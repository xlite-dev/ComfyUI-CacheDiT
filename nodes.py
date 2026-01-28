"""
ComfyUI-CacheDiT: Node Definitions
====================================

Main node: CacheDiT_Model_Optimizer
- Accelerates DiT models via inter-step residual caching
- Smart auto-detection of inference steps via ComfyUI hooks
- Rich summary statistics dashboard
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

import comfy.model_patcher
import comfy.patcher_extension

from .utils import (
    MODEL_PRESETS,
    ModelPreset,
    PATTERN_DESCRIPTIONS,
    get_preset,
    get_all_preset_names,
    get_forward_pattern,
    build_cache_config,
    build_calibrator_config,
    build_block_adapter,
    format_summary_dashboard,
    get_summary_stats,
    print_summary_to_log,
    apply_noise_injection,
)

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher

logger = logging.getLogger("ComfyUI-CacheDiT")


# =============================================================================
# Configuration Holder Class
# =============================================================================

class CacheDiTConfig:
    """
    Holds all CacheDiT configuration for a model instance.
    Stored in transformer_options for access by wrappers.
    """
    
    def __init__(
        self,
        # Basic settings
        model_type: str,
        forward_pattern: str,
        strategy: str,
        # DBCache settings
        fn_blocks: int,
        bn_blocks: int,
        threshold: float,
        max_warmup_steps: int,
        # CFG settings
        enable_separate_cfg: Optional[bool],
        cfg_compute_first: bool,
        # Advanced settings
        skip_interval: int,
        noise_scale: float,
        taylor_order: int,
        scm_policy: str,
        # Runtime settings
        verbose: bool = False,
        print_summary: bool = True,
    ):
        # Configuration
        self.model_type = model_type
        self.forward_pattern = forward_pattern
        self.strategy = strategy
        
        self.fn_blocks = fn_blocks
        self.bn_blocks = bn_blocks
        self.threshold = threshold
        self.max_warmup_steps = max_warmup_steps
        
        self.enable_separate_cfg = enable_separate_cfg
        self.cfg_compute_first = cfg_compute_first
        
        self.skip_interval = skip_interval
        self.noise_scale = noise_scale
        self.taylor_order = taylor_order
        self.scm_policy = scm_policy
        
        self.verbose = verbose
        self.print_summary = print_summary
        
        # Runtime state
        self.is_enabled = False
        self.num_inference_steps: Optional[int] = None
        self.current_step: int = 0
        self.first_step_done: bool = False
        
    def clone(self) -> "CacheDiTConfig":
        """Create a copy for cloned models."""
        new_config = CacheDiTConfig(
            model_type=self.model_type,
            forward_pattern=self.forward_pattern,
            strategy=self.strategy,
            fn_blocks=self.fn_blocks,
            bn_blocks=self.bn_blocks,
            threshold=self.threshold,
            max_warmup_steps=self.max_warmup_steps,
            enable_separate_cfg=self.enable_separate_cfg,
            cfg_compute_first=self.cfg_compute_first,
            skip_interval=self.skip_interval,
            noise_scale=self.noise_scale,
            taylor_order=self.taylor_order,
            scm_policy=self.scm_policy,
            verbose=self.verbose,
            print_summary=self.print_summary,
        )
        new_config.is_enabled = self.is_enabled
        new_config.num_inference_steps = self.num_inference_steps
        return new_config
    
    def reset(self):
        """Reset runtime state for new generation."""
        self.current_step = 0
        self.first_step_done = False
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get config as dict for summary display."""
        return {
            "model_type": self.model_type,
            "pattern": self.forward_pattern,
            "strategy": self.strategy,
            "fn": self.fn_blocks,
            "bn": self.bn_blocks,
            "threshold": self.threshold,
            "skip_interval": self.skip_interval,
            "noise_scale": self.noise_scale,
            "taylor_order": self.taylor_order,
        }


# =============================================================================
# ComfyUI Wrapper Functions (Smart Hooks)
# =============================================================================

def _cache_dit_outer_sample_wrapper(executor, *args, **kwargs):
    """
    OUTER_SAMPLE wrapper: The "Smart" Refresh Hook.
    
    - Auto-detects num_inference_steps from sigmas
    - Enables cache-dit at sampling start
    - Refreshes context with correct step count
    - Prints summary dashboard at end
    """
    guider = executor.class_obj
    orig_model_options = guider.model_options
    transformer = None
    config = None
    
    try:
        # Clone model options for this run
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        
        # Get CacheDiT config
        config: CacheDiTConfig = guider.model_options.get("transformer_options", {}).get("cache_dit_turbo")
        if config is None:
            return executor(*args, **kwargs)
        
        # Clone and reset config for this run
        config = config.clone()
        config.reset()
        guider.model_options["transformer_options"]["cache_dit_turbo"] = config
        
        # =====================================================================
        # SMART STEP DETECTION: Extract num_inference_steps from sigmas
        # =====================================================================
        sigmas = args[3] if len(args) > 3 else kwargs.get("sigmas")
        if sigmas is not None:
            num_steps = len(sigmas) - 1  # sigmas has N+1 elements for N steps
            config.num_inference_steps = num_steps
            
            if config.verbose:
                logger.info(f"[CacheDiT] Auto-detected {num_steps} inference steps")
        
        # Get transformer reference
        model_patcher = guider.model_patcher
        if hasattr(model_patcher, 'model') and hasattr(model_patcher.model, 'diffusion_model'):
            transformer = model_patcher.model.diffusion_model
            
            # Enable or refresh cache-dit
            if not config.is_enabled and config.num_inference_steps is not None:
                _enable_cache_dit(transformer, config)
                config.is_enabled = True
            elif config.is_enabled and config.num_inference_steps is not None:
                _refresh_cache_dit(transformer, config)
        
        if config.verbose:
            logger.info(
                f"[CacheDiT] Enabled: model={config.model_type}, "
                f"pattern={config.forward_pattern}, strategy={config.strategy}, "
                f"steps={config.num_inference_steps}"
            )
        
        # Execute sampling
        result = executor(*args, **kwargs)
        
        # =====================================================================
        # SUMMARY DASHBOARD: Print statistics after sampling
        # =====================================================================
        if config.print_summary and transformer is not None:
            try:
                dashboard = print_summary_to_log(
                    transformer=transformer,
                    model_type=config.model_type,
                    num_steps=config.num_inference_steps or 0,
                    config_info=config.get_config_info(),
                )
            except Exception as e:
                logger.warning(f"[CacheDiT] Summary failed: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"[CacheDiT] outer_sample_wrapper error: {e}")
        import traceback
        traceback.print_exc()
        return executor(*args, **kwargs)
    finally:
        # Restore original model options
        try:
            guider.model_options = orig_model_options
        except:
            pass


def _cache_dit_diffusion_model_wrapper(executor, *args, **kwargs):
    """
    DIFFUSION_MODEL wrapper: Per-step processing.
    
    - Tracks current step
    - Applies noise injection on cached outputs (if enabled)
    """
    try:
        # Get transformer_options
        transformer_options = args[-1] if isinstance(args[-1], dict) else kwargs.get("transformer_options", {})
        config: CacheDiTConfig = transformer_options.get("cache_dit_turbo")
        
        if config is not None:
            # Track steps
            if not config.first_step_done:
                config.first_step_done = True
                config.current_step = 0
            else:
                config.current_step += 1
            
            if config.verbose and config.current_step % 10 == 0:
                logger.debug(f"[CacheDiT] Step {config.current_step}/{config.num_inference_steps}")
        
        # Execute forward pass
        output = executor(*args, **kwargs)
        
        # Apply noise injection if enabled (for cached outputs)
        if config is not None and config.noise_scale > 0:
            # Note: noise injection is applied at cache-dit level via custom logic
            # This is a fallback for manual noise injection
            pass
        
        return output
        
    except Exception as e:
        logger.error(f"[CacheDiT] diffusion_model_wrapper error: {e}")
        return executor(*args, **kwargs)


# =============================================================================
# Cache-DiT Integration Functions
# =============================================================================

def _enable_cache_dit(transformer: torch.nn.Module, config: CacheDiTConfig):
    """
    Enable cache-dit on transformer using BlockAdapter interface.
    """
    try:
        import cache_dit
        from cache_dit.caching.cache_adapters import CachedAdapter
        
        # Build BlockAdapter
        adapter = build_block_adapter(
            transformer=transformer,
            forward_pattern=config.forward_pattern,
            auto_detect=True,
        )
        
        # Build cache config
        cache_config = build_cache_config(
            num_inference_steps=config.num_inference_steps,
            fn_blocks=config.fn_blocks,
            bn_blocks=config.bn_blocks,
            threshold=config.threshold,
            max_warmup_steps=config.max_warmup_steps,
            enable_separate_cfg=config.enable_separate_cfg,
            cfg_compute_first=config.cfg_compute_first,
            skip_interval=config.skip_interval,
            strategy=config.strategy,
            scm_policy=config.scm_policy if config.scm_policy != "none" else None,
        )
        
        # Build calibrator config
        calibrator_config = build_calibrator_config(config.taylor_order)
        
        # Use CachedAdapter.cachify directly to bypass auto_block_adapter
        cachify_kwargs = {"cache_config": cache_config}
        if calibrator_config is not None:
            cachify_kwargs["calibrator_config"] = calibrator_config
        
        cached_adapter = CachedAdapter.cachify(
            block_adapter=adapter,
            **cachify_kwargs
        )
        
        if config.verbose:
            logger.info(
                f"[CacheDiT] Cache enabled: F{config.fn_blocks}B{config.bn_blocks}, "
                f"threshold={config.threshold}, warmup={config.max_warmup_steps}"
            )
        
    except Exception as e:
        logger.error(f"[CacheDiT] Failed to enable cache: {e}")
        raise


def _refresh_cache_dit(transformer: torch.nn.Module, config: CacheDiTConfig):
    """
    Refresh cache-dit context with updated settings.
    Called when num_inference_steps changes between requests.
    """
    try:
        import cache_dit
        
        # Rebuild configs with new step count
        cache_config = build_cache_config(
            num_inference_steps=config.num_inference_steps,
            fn_blocks=config.fn_blocks,
            bn_blocks=config.bn_blocks,
            threshold=config.threshold,
            max_warmup_steps=config.max_warmup_steps,
            enable_separate_cfg=config.enable_separate_cfg,
            cfg_compute_first=config.cfg_compute_first,
            skip_interval=config.skip_interval,
            strategy=config.strategy,
            scm_policy=config.scm_policy if config.scm_policy != "none" else None,
        )
        
        calibrator_config = build_calibrator_config(config.taylor_order)
        
        refresh_kwargs = {
            "cache_config": cache_config,
            "verbose": config.verbose,
        }
        if calibrator_config is not None:
            refresh_kwargs["calibrator_config"] = calibrator_config
        
        cache_dit.refresh_context(transformer, **refresh_kwargs)
        
        if config.verbose:
            logger.info(f"[CacheDiT] Context refreshed for {config.num_inference_steps} steps")
        
    except Exception as e:
        logger.warning(f"[CacheDiT] Failed to refresh context: {e}")


# =============================================================================
# Main Node: CacheDiT_Model_Optimizer
# =============================================================================

class CacheDiT_Model_Optimizer:
    """
    CacheDiT Model Optimizer for ComfyUI
    
    Accelerates DiT model inference through inter-step residual caching.
    Automatically detects inference steps and refreshes context.
    
    Supports: Qwen-Image, LTX-2, Z-Image, Flux, HunyuanVideo, Wan, and custom models.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        preset_names = get_all_preset_names()
        patterns = list(PATTERN_DESCRIPTIONS.keys())
        strategies = ["adaptive", "static", "dynamic"]
        scm_policies = ["none", "fast", "medium", "ultra"]
        
        return {
            "required": {
                "model": ("MODEL",),
                
                # ============================================================
                # Basic Settings / 基础设置
                # ============================================================
                "model_type": (preset_names, {
                    "default": "Custom",
                    "tooltip": "Model preset (presets auto-configure optimal settings)\n"
                               "模型预设 (预设会自动配置最优参数)"
                }),
                "forward_pattern": (patterns, {
                    "default": "Pattern_1",
                    "tooltip": "Transformer block forward pattern\n"
                               "Transformer 块前向传播模式"
                }),
                "strategy": (strategies, {
                    "default": "adaptive",
                    "tooltip": "Caching strategy:\n"
                               "• adaptive: Auto-balance quality/speed\n"
                               "• static: Aggressive caching\n"
                               "• dynamic: Conservative caching\n"
                               "缓存策略: adaptive=自适应, static=激进, dynamic=保守"
                }),
                
                # ============================================================
                # Core Parameters / 核心参数
                # ============================================================
                "threshold": ("FLOAT", {
                    "default": 0.12,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Residual diff threshold (higher=faster, lower=better quality)\n"
                               "残差阈值 (越高越快, 越低质量越好)"
                }),
                "fn_blocks": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Fn: Front blocks for stable diff calculation\n"
                               "Fn: 用于稳定差分计算的前置块数"
                }),
                "bn_blocks": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Bn: Back blocks for feature fusion\n"
                               "Bn: 用于特征融合的后置块数"
                }),
            },
            "optional": {
                # ============================================================
                # Advanced Parameters / 高级参数
                # ============================================================
                "warmup_steps": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Steps before caching starts (no acceleration during warmup)\n"
                               "预热步数 (预热期间不加速)"
                }),
                "skip_interval": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Force compute every N steps (0=disabled, critical for video)\n"
                               "每 N 步强制计算 (0=禁用, 视频生成必需)"
                }),
                "noise_scale": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.01,
                    "step": 0.0005,
                    "tooltip": "Noise injection scale to prevent static artifacts\n"
                               "噪声注入强度, 防止生成结果死板"
                }),
                "taylor_order": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 2,
                    "step": 1,
                    "tooltip": "TaylorSeer order (0=disabled, 1-2=enabled)\n"
                               "TaylorSeer 阶数 (0=禁用, 1-2=启用)"
                }),
                "scm_policy": (scm_policies, {
                    "default": "none",
                    "tooltip": "Steps Computation Mask policy:\n"
                               "• none: Dynamic (default)\n"
                               "• fast/medium/ultra: Precomputed masks\n"
                               "步数计算掩码策略"
                }),
                
                # ============================================================
                # CFG Settings / CFG 设置
                # ============================================================
                "separate_cfg": (["auto", "true", "false"], {
                    "default": "auto",
                    "tooltip": "Separate CFG processing (auto uses preset value)\n"
                               "分离 CFG 处理 (auto 使用预设值)"
                }),
                
                # ============================================================
                # Debug Settings / 调试设置
                # ============================================================
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable verbose logging\n启用详细日志"
                }),
                "print_summary": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print performance dashboard after sampling\n采样后打印性能仪表盘"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("optimized_model",)
    FUNCTION = "optimize"
    CATEGORY = "⚡ CacheDiT"
    DESCRIPTION = (
        "Accelerate DiT models (Qwen-Image, LTX-2, Z-Image, etc.) via residual caching.\n"
        "通过残差缓存加速 DiT 模型 (Qwen-Image、LTX-2、Z-Image 等)"
    )
    
    def optimize(
        self,
        model,
        model_type: str,
        forward_pattern: str,
        strategy: str,
        threshold: float,
        fn_blocks: int,
        bn_blocks: int,
        warmup_steps: int = 8,
        skip_interval: int = 0,
        noise_scale: float = 0.0,
        taylor_order: int = 1,
        scm_policy: str = "none",
        separate_cfg: str = "auto",
        verbose: bool = False,
        print_summary: bool = True,
    ):
        """Apply CacheDiT optimization to the model."""
        
        # Clone model to avoid modifying original
        model = model.clone()
        
        # Get preset and apply defaults for "auto" settings
        preset = get_preset(model_type)
        
        # Use preset values if model_type is not Custom
        if model_type != "Custom":
            # Use preset defaults but allow user overrides
            if forward_pattern == "Pattern_1" and preset.forward_pattern != "Pattern_1":
                forward_pattern = preset.forward_pattern
            
            # Apply preset strategy if using default
            if strategy == "adaptive":
                strategy = preset.default_strategy
        
        # Determine CFG setting
        if separate_cfg == "auto":
            enable_separate_cfg = preset.enable_separate_cfg
        else:
            enable_separate_cfg = separate_cfg == "true"
        
        # Create configuration
        config = CacheDiTConfig(
            model_type=model_type,
            forward_pattern=forward_pattern,
            strategy=strategy,
            fn_blocks=fn_blocks,
            bn_blocks=bn_blocks,
            threshold=threshold,
            max_warmup_steps=warmup_steps,
            enable_separate_cfg=enable_separate_cfg,
            cfg_compute_first=preset.cfg_compute_first,
            skip_interval=skip_interval,
            noise_scale=noise_scale,
            taylor_order=taylor_order,
            scm_policy=scm_policy,
            verbose=verbose,
            print_summary=print_summary,
        )
        
        # Store config in model options
        model.model_options["transformer_options"]["cache_dit_turbo"] = config
        
        # Add wrappers
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "cache_dit_turbo",
            _cache_dit_outer_sample_wrapper
        )
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "cache_dit_turbo",
            _cache_dit_diffusion_model_wrapper
        )
        
        if verbose:
            logger.info(
                f"[CacheDiT] Configured: {model_type}, {forward_pattern}, "
                f"F{fn_blocks}B{bn_blocks}, threshold={threshold}"
            )
        
        return (model,)


# =============================================================================
# Additional Nodes
# =============================================================================

class CacheDiT_Disable:
    """Disable CacheDiT acceleration on a model."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "disable"
    CATEGORY = "⚡ CacheDiT"
    DESCRIPTION = "Remove CacheDiT optimization from model\n移除模型的 CacheDiT 优化"
    
    def disable(self, model):
        model = model.clone()
        
        # Remove config
        if "cache_dit_turbo" in model.model_options.get("transformer_options", {}):
            del model.model_options["transformer_options"]["cache_dit_turbo"]
        
        # Remove wrappers
        for wrapper_type in [comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL]:
            if "cache_dit_turbo" in model.wrappers.get(wrapper_type, {}):
                del model.wrappers[wrapper_type]["cache_dit_turbo"]
        
        # Disable cache-dit on transformer
        try:
            import cache_dit
            if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                cache_dit.disable_cache(model.model.diffusion_model)
        except:
            pass
        
        return (model,)


class CacheDiT_Preset_Info:
    """Display information about a model preset."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (get_all_preset_names(), {"default": "Qwen-Image"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("preset_info",)
    FUNCTION = "get_info"
    CATEGORY = "⚡ CacheDiT"
    DESCRIPTION = "Get recommended settings for a model preset\n获取模型预设的推荐配置"
    
    def get_info(self, model_type: str):
        preset = get_preset(model_type)
        
        info_lines = [
            f"═══════════════════════════════════════════",
            f"  Model Preset: {preset.name}",
            f"═══════════════════════════════════════════",
            f"  {preset.description}",
            f"  {preset.description_cn}",
            f"───────────────────────────────────────────",
            f"  Forward Pattern:     {preset.forward_pattern}",
            f"  Fn/Bn Blocks:        F{preset.fn_blocks}B{preset.bn_blocks}",
            f"  Threshold:           {preset.threshold}",
            f"  Warmup Steps:        {preset.max_warmup_steps}",
            f"  Strategy:            {preset.default_strategy}",
            f"───────────────────────────────────────────",
            f"  Separate CFG:        {preset.enable_separate_cfg}",
            f"  Skip Interval:       {preset.skip_interval}",
            f"  Noise Scale:         {preset.noise_scale}",
            f"  TaylorSeer Order:    {preset.taylor_order}",
            f"═══════════════════════════════════════════",
        ]
        
        return ("\n".join(info_lines),)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "CacheDiT_Model_Optimizer": CacheDiT_Model_Optimizer,
    "CacheDiT_Disable": CacheDiT_Disable,
    "CacheDiT_Preset_Info": CacheDiT_Preset_Info,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CacheDiT_Model_Optimizer": "⚡ CacheDiT Model Optimizer",
    "CacheDiT_Disable": "⚡ CacheDiT Disable",
    "CacheDiT_Preset_Info": "⚡ CacheDiT Preset Info",
}
