"""ComfyUI-CacheDiT: Utility Functions
=====================================

This module provides:
- Model preset configurations
- BlockAdapter construction
- Cache configuration builders
- Summary statistics formatting (ASCII dashboard)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger("ComfyUI-CacheDiT")


# =============================================================================
# Model Presets - Hardcoded Recommended Configurations (2026 Models)
# =============================================================================

@dataclass
class ModelPreset:
    """Preset configuration for a specific model type."""
    name: str
    description: str
    description_cn: str
    # Forward pattern
    forward_pattern: str
    # DBCache config
    fn_blocks: int  # Fn_compute_blocks
    bn_blocks: int  # Bn_compute_blocks
    threshold: float  # residual_diff_threshold
    max_warmup_steps: int
    # CFG settings
    enable_separate_cfg: Optional[bool]
    cfg_compute_first: bool = False
    # Advanced settings
    skip_interval: int = 0  # Force compute every N steps (0=disabled)
    noise_scale: float = 0.0  # Noise injection scale
    # Strategy
    default_strategy: str = "adaptive"
    # TaylorSeer
    taylor_order: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "forward_pattern": self.forward_pattern,
            "fn_blocks": self.fn_blocks,
            "bn_blocks": self.bn_blocks,
            "threshold": self.threshold,
            "max_warmup_steps": self.max_warmup_steps,
            "enable_separate_cfg": self.enable_separate_cfg,
            "cfg_compute_first": self.cfg_compute_first,
            "skip_interval": self.skip_interval,
            "noise_scale": self.noise_scale,
            "default_strategy": self.default_strategy,
            "taylor_order": self.taylor_order,
        }


# Hardcoded presets for 2026 mainstream models
MODEL_PRESETS: Dict[str, ModelPreset] = {
    # =========================================================================
    # Qwen-Image Series
    # =========================================================================
    "Qwen-Image": ModelPreset(
        name="Qwen-Image",
        description="Qwen-Image standard (2511/2512 edit)",
        description_cn="Qwen-Image 标准版 (2511/2512 编辑)",
        forward_pattern="Pattern_1",
        fn_blocks=1,  # F1B0 as specified
        bn_blocks=0,
        threshold=0.12,
        max_warmup_steps=8,
        enable_separate_cfg=True,
        cfg_compute_first=False,
        skip_interval=0,
        noise_scale=0.0,
        default_strategy="adaptive",
        taylor_order=1,
    ),
    "Qwen-Image-Layered": ModelPreset(
        name="Qwen-Image-Layered",
        description="Qwen-Image Layered (Alpha layer protection)",
        description_cn="Qwen-Image 分层版 (保护 Alpha 层)",
        forward_pattern="Pattern_1",
        fn_blocks=8,  # F8B4 to protect Alpha layer
        bn_blocks=4,
        threshold=0.10,
        max_warmup_steps=10,
        enable_separate_cfg=True,
        cfg_compute_first=False,
        skip_interval=0,
        noise_scale=0.0,
        default_strategy="adaptive",
        taylor_order=1,
    ),
    
    # =========================================================================
    # LTX-2 Video Series
    # =========================================================================
    "LTX-2-T2V": ModelPreset(
        name="LTX-2-T2V",
        description="LTX-2 Text-to-Video (temporal consistency)",
        description_cn="LTX-2 文生视频 (时序一致性优化)",
        forward_pattern="Pattern_1",
        fn_blocks=4,  # F4B4 for video
        bn_blocks=4,
        threshold=0.08,
        max_warmup_steps=6,
        enable_separate_cfg=False,
        cfg_compute_first=False,
        skip_interval=3,  # Force compute every 3 steps for temporal consistency
        noise_scale=0.001,
        default_strategy="dynamic",
        taylor_order=1,
    ),
    "LTX-2-I2V": ModelPreset(
        name="LTX-2-I2V",
        description="LTX-2 Image-to-Video",
        description_cn="LTX-2 图生视频",
        forward_pattern="Pattern_1",
        fn_blocks=4,
        bn_blocks=4,
        threshold=0.08,
        max_warmup_steps=6,
        enable_separate_cfg=False,
        cfg_compute_first=False,
        skip_interval=3,
        noise_scale=0.001,
        default_strategy="dynamic",
        taylor_order=1,
    ),
    
    # =========================================================================
    # Z-Image Series
    # =========================================================================
    "Z-Image": ModelPreset(
        name="Z-Image",
        description="Z-Image standard",
        description_cn="Z-Image 标准版",
        forward_pattern="Pattern_1",
        fn_blocks=8,  # F8B0 as specified
        bn_blocks=0,
        threshold=0.12,
        max_warmup_steps=8,
        enable_separate_cfg=True,
        cfg_compute_first=False,
        skip_interval=0,
        noise_scale=0.0015,  # Small noise injection
        default_strategy="adaptive",
        taylor_order=1,
    ),
    "Z-Image-Turbo": ModelPreset(
        name="Z-Image-Turbo",
        description="Z-Image Turbo (distilled, 4-9 steps)",
        description_cn="Z-Image Turbo (蒸馏版, 4-9步)",
        forward_pattern="Pattern_1",
        fn_blocks=4,
        bn_blocks=0,
        threshold=0.15,
        max_warmup_steps=3,
        enable_separate_cfg=True,
        cfg_compute_first=False,
        skip_interval=0,
        noise_scale=0.002,
        default_strategy="static",
        taylor_order=0,  # Disabled for low-step models
    ),
    
    # =========================================================================
    # Other Popular Models
    # =========================================================================
    "Flux": ModelPreset(
        name="Flux",
        description="Flux.1 Dev/Schnell",
        description_cn="Flux.1 开发版/快速版",
        forward_pattern="Pattern_0",
        fn_blocks=10,
        bn_blocks=0,
        threshold=0.10,
        max_warmup_steps=8,
        enable_separate_cfg=True,
        cfg_compute_first=False,
        skip_interval=0,
        noise_scale=0.0,
        default_strategy="adaptive",
        taylor_order=1,
    ),
    "HunyuanVideo": ModelPreset(
        name="HunyuanVideo",
        description="Hunyuan Video (fused CFG)",
        description_cn="混元视频 (融合 CFG)",
        forward_pattern="Pattern_3",
        fn_blocks=6,
        bn_blocks=2,
        threshold=0.08,
        max_warmup_steps=8,
        enable_separate_cfg=False,  # Fused CFG
        cfg_compute_first=False,
        skip_interval=2,
        noise_scale=0.001,
        default_strategy="dynamic",
        taylor_order=1,
    ),
    "Wan-2.1": ModelPreset(
        name="Wan-2.1",
        description="Wan 2.1 Video",
        description_cn="万相 2.1 视频",
        forward_pattern="Pattern_3",
        fn_blocks=6,
        bn_blocks=2,
        threshold=0.08,
        max_warmup_steps=8,
        enable_separate_cfg=True,
        cfg_compute_first=False,
        skip_interval=2,
        noise_scale=0.001,
        default_strategy="dynamic",
        taylor_order=1,
    ),
    
    # =========================================================================
    # Custom / Fallback
    # =========================================================================
    "Custom": ModelPreset(
        name="Custom",
        description="Custom model (manual configuration)",
        description_cn="自定义模型 (手动配置)",
        forward_pattern="Pattern_1",
        fn_blocks=8,
        bn_blocks=0,
        threshold=0.12,
        max_warmup_steps=8,
        enable_separate_cfg=None,
        cfg_compute_first=False,
        skip_interval=0,
        noise_scale=0.0,
        default_strategy="adaptive",
        taylor_order=1,
    ),
}


def get_preset(model_type: str) -> ModelPreset:
    """Get preset configuration for a model type."""
    return MODEL_PRESETS.get(model_type, MODEL_PRESETS["Custom"])


def get_all_preset_names() -> List[str]:
    """Get list of all available preset names."""
    return list(MODEL_PRESETS.keys())


# =============================================================================
# Forward Pattern Utilities
# =============================================================================

PATTERN_DESCRIPTIONS = {
    "Pattern_0": "Return_H_First=True, In=(h,enc_h), Out=(h,enc_h) - Flux style",
    "Pattern_1": "Return_H_First=False, In=(h,enc_h), Out=(enc_h,h) - Qwen/LTX/Z-Image",
    "Pattern_2": "Return_H_Only=True, In=(h,enc_h), Out=(h,) - Single output",
    "Pattern_3": "Forward_H_only=True, In=(h,), Out=(h,) - Hunyuan/Wan",
    "Pattern_4": "Return_H_First=True, In=(h,), Out=(h,enc_h) - Special",
    "Pattern_5": "Return_H_First=False, In=(h,), Out=(enc_h,h) - Special",
}


def get_forward_pattern(pattern_name: str):
    """Get ForwardPattern enum from cache_dit."""
    try:
        import cache_dit
        pattern_map = {
            "Pattern_0": cache_dit.ForwardPattern.Pattern_0,
            "Pattern_1": cache_dit.ForwardPattern.Pattern_1,
            "Pattern_2": cache_dit.ForwardPattern.Pattern_2,
            "Pattern_3": cache_dit.ForwardPattern.Pattern_3,
            "Pattern_4": cache_dit.ForwardPattern.Pattern_4,
            "Pattern_5": cache_dit.ForwardPattern.Pattern_5,
        }
        return pattern_map.get(pattern_name, cache_dit.ForwardPattern.Pattern_1)
    except ImportError:
        raise ImportError(
            "cache_dit library not found. Please install: pip install cache-dit>=1.2.0"
        )


# =============================================================================
# Cache Configuration Builders
# =============================================================================

def build_cache_config(
    num_inference_steps: Optional[int],
    fn_blocks: int,
    bn_blocks: int,
    threshold: float,
    max_warmup_steps: int,
    enable_separate_cfg: Optional[bool],
    cfg_compute_first: bool,
    skip_interval: int,
    strategy: str,
    scm_policy: Optional[str] = None,
):
    """
    Build DBCacheConfig with advanced settings.
    
    Args:
        num_inference_steps: Total inference steps (None for unknown)
        fn_blocks: Fn_compute_blocks
        bn_blocks: Bn_compute_blocks  
        threshold: residual_diff_threshold
        max_warmup_steps: Steps before caching starts
        enable_separate_cfg: CFG separation mode
        cfg_compute_first: Compute CFG first
        skip_interval: Force compute every N steps (0=disabled)
        strategy: 'adaptive', 'static', or 'dynamic'
        scm_policy: Steps computation mask policy
    """
    try:
        import cache_dit
        from cache_dit import DBCacheConfig, steps_mask
        
        config = DBCacheConfig(
            Fn_compute_blocks=fn_blocks,
            Bn_compute_blocks=bn_blocks,
            residual_diff_threshold=threshold,
            max_warmup_steps=max_warmup_steps,
            num_inference_steps=num_inference_steps,
        )
        
        # CFG settings
        if enable_separate_cfg is not None:
            config.enable_separate_cfg = enable_separate_cfg
        config.cfg_compute_first = cfg_compute_first
        
        # Strategy-based configuration
        if strategy == "static":
            # Static: More aggressive caching, fewer compute steps
            config.max_cached_steps = -1
            config.max_continuous_cached_steps = -1
        elif strategy == "dynamic":
            # Dynamic: Conservative caching
            config.max_cached_steps = -1
            config.max_continuous_cached_steps = 4  # Limit continuous caching
        else:  # adaptive (default)
            config.max_cached_steps = -1
            config.max_continuous_cached_steps = -1
        
        # Apply SCM policy or skip_interval
        if num_inference_steps is not None:
            if scm_policy and scm_policy != "none":
                # Use predefined SCM policy
                scm_mask = steps_mask(
                    total_steps=num_inference_steps,
                    mask_policy=scm_policy,
                )
                config.steps_computation_mask = scm_mask
                config.steps_computation_policy = "dynamic"
            elif skip_interval > 0:
                # Generate custom mask with skip_interval
                scm_mask = _generate_skip_interval_mask(
                    num_inference_steps, skip_interval, max_warmup_steps
                )
                config.steps_computation_mask = scm_mask
                config.steps_computation_policy = "dynamic"
        
        return config
        
    except ImportError as e:
        raise ImportError(f"Failed to build cache config: {e}")


def _generate_skip_interval_mask(
    total_steps: int, 
    skip_interval: int, 
    warmup_steps: int
) -> List[int]:
    """
    Generate steps computation mask with skip_interval.
    
    Forces computation every skip_interval steps for temporal consistency.
    
    Example with total_steps=20, skip_interval=3, warmup=4:
    [1,1,1,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 1]
    """
    mask = []
    
    for step in range(total_steps):
        if step < warmup_steps:
            # Warmup: always compute
            mask.append(1)
        elif step == total_steps - 1:
            # Last step: always compute
            mask.append(1)
        elif (step - warmup_steps) % skip_interval == 0:
            # Force compute at interval
            mask.append(1)
        else:
            # Cache
            mask.append(0)
    
    return mask


def build_calibrator_config(taylor_order: int):
    """Build TaylorSeerCalibratorConfig if taylor_order > 0."""
    if taylor_order <= 0:
        return None
    
    try:
        from cache_dit import TaylorSeerCalibratorConfig
        
        return TaylorSeerCalibratorConfig(
            enable_calibrator=True,
            enable_encoder_calibrator=True,
            taylorseer_order=taylor_order,
        )
    except ImportError:
        logger.warning("TaylorSeerCalibratorConfig not available")
        return None


# =============================================================================
# BlockAdapter Construction
# =============================================================================

def build_block_adapter(
    transformer: torch.nn.Module,
    forward_pattern: str,
    auto_detect: bool = True,
):
    """
    Build BlockAdapter for a transformer model.
    
    Args:
        transformer: The diffusion model transformer
        forward_pattern: Pattern name (Pattern_0 to Pattern_5)
        auto_detect: Auto-detect transformer blocks
    """
    try:
        from cache_dit import BlockAdapter
        
        pattern = get_forward_pattern(forward_pattern)
        
        # For ComfyUI custom models (non-diffusers), manually construct BlockAdapter
        # Extract transformer blocks manually to avoid pipeline detection
        if auto_detect:
            # Auto-detect blocks from transformer
            blocks = []
            for name, module in transformer.named_children():
                # Common block names in DiT models
                if any(keyword in name.lower() for keyword in ['block', 'layer', 'transformer_block']):
                    blocks.append(module)
            
            if not blocks:
                # Fallback: try to find blocks in nested structures
                for module in transformer.modules():
                    if hasattr(module, '__iter__') and not isinstance(module, torch.nn.Sequential):
                        continue
                    if isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
                        blocks = list(module)
                        break
        
            if blocks:
                adapter = BlockAdapter(
                    blocks=blocks,
                    forward_pattern=pattern,
                )
                return adapter
        
        # If auto-detect failed or not requested, try direct constructor
        adapter = BlockAdapter(
            transformer=transformer,
            forward_pattern=pattern,
            auto=auto_detect,
        )
        
        return adapter
        
    except Exception as e:
        raise RuntimeError(f"Failed to build BlockAdapter: {e}")


# =============================================================================
# Summary Statistics & ASCII Dashboard
# =============================================================================

def format_summary_dashboard(
    stats: Dict[str, Any],
    model_type: str,
    num_steps: int,
    config_info: Dict[str, Any],
) -> str:
    """
    Format cache-dit summary statistics as an ASCII dashboard.
    
    Returns a beautifully formatted string for terminal/log output.
    """
    if not stats:
        return "╔══════════════════════════════════════════════════════════════╗\n║  CacheDiT Summary: No statistics available                   ║\n╚══════════════════════════════════════════════════════════════╝"
    
    # Extract key metrics
    total_steps = stats.get("total_steps", num_steps)
    cached_steps = stats.get("cached_steps", 0)
    computed_steps = stats.get("computed_steps", total_steps - cached_steps)
    cache_hit_rate = (cached_steps / max(total_steps, 1)) * 100
    
    avg_diff = stats.get("avg_residual_diff", 0.0)
    max_diff = stats.get("max_residual_diff", 0.0)
    speedup = stats.get("speedup", total_steps / max(computed_steps, 1))
    
    # Build ASCII table
    width = 66
    lines = []
    
    # Header
    lines.append("╔" + "═" * (width - 2) + "╗")
    title = "CacheDiT Performance Dashboard"
    lines.append("║" + title.center(width - 2) + "║")
    lines.append("╠" + "═" * (width - 2) + "╣")
    
    # Model info section
    lines.append("║" + f"  Model: {model_type}".ljust(width - 2) + "║")
    lines.append("║" + f"  Pattern: {config_info.get('pattern', 'N/A')}".ljust(width - 2) + "║")
    lines.append("║" + f"  Strategy: {config_info.get('strategy', 'N/A')}".ljust(width - 2) + "║")
    lines.append("╠" + "─" * (width - 2) + "╣")
    
    # Performance metrics
    lines.append("║" + "  📊 Performance Metrics".ljust(width - 2) + "║")
    lines.append("║" + "─" * (width - 4) + "  ║")
    
    # Create metric rows
    metrics = [
        ("Total Steps", f"{total_steps}"),
        ("Computed Steps", f"{computed_steps}"),
        ("Cached Steps", f"{cached_steps}"),
        ("Cache Hit Rate", f"{cache_hit_rate:.1f}%"),
        ("Estimated Speedup", f"{speedup:.2f}x"),
    ]
    
    for label, value in metrics:
        row = f"  {label}:".ljust(25) + f"{value}".rjust(width - 29)
        lines.append("║" + row + "║")
    
    lines.append("╠" + "─" * (width - 2) + "╣")
    
    # Quality metrics
    lines.append("║" + "  🎯 Quality Metrics".ljust(width - 2) + "║")
    lines.append("║" + "─" * (width - 4) + "  ║")
    
    quality_metrics = [
        ("Threshold", f"{config_info.get('threshold', 0):.4f}"),
        ("Avg Residual Diff", f"{avg_diff:.6f}"),
        ("Max Residual Diff", f"{max_diff:.6f}"),
        ("Fn/Bn Blocks", f"F{config_info.get('fn', 0)}B{config_info.get('bn', 0)}"),
    ]
    
    for label, value in quality_metrics:
        row = f"  {label}:".ljust(25) + f"{value}".rjust(width - 29)
        lines.append("║" + row + "║")
    
    # Advanced settings if present
    if config_info.get("skip_interval", 0) > 0 or config_info.get("noise_scale", 0) > 0:
        lines.append("╠" + "─" * (width - 2) + "╣")
        lines.append("║" + "  ⚙️  Advanced Settings".ljust(width - 2) + "║")
        lines.append("║" + "─" * (width - 4) + "  ║")
        
        if config_info.get("skip_interval", 0) > 0:
            row = f"  Skip Interval:".ljust(25) + f"{config_info['skip_interval']}".rjust(width - 29)
            lines.append("║" + row + "║")
        if config_info.get("noise_scale", 0) > 0:
            row = f"  Noise Scale:".ljust(25) + f"{config_info['noise_scale']:.6f}".rjust(width - 29)
            lines.append("║" + row + "║")
        if config_info.get("taylor_order", 0) > 0:
            row = f"  TaylorSeer Order:".ljust(25) + f"{config_info['taylor_order']}".rjust(width - 29)
            lines.append("║" + row + "║")
    
    # Speedup visualization bar
    lines.append("╠" + "─" * (width - 2) + "╣")
    lines.append("║" + "  🚀 Speedup Visualization".ljust(width - 2) + "║")
    
    bar_width = width - 12
    filled = int((speedup - 1.0) / 2.0 * bar_width)  # Scale 1x-3x to bar
    filled = max(0, min(filled, bar_width))
    bar = "█" * filled + "░" * (bar_width - filled)
    lines.append("║" + f"  [{bar}]  ║")
    lines.append("║" + f"  1.0x".ljust(bar_width // 2) + f"3.0x".rjust(bar_width // 2 + 6) + "  ║")
    
    # Footer
    lines.append("╠" + "═" * (width - 2) + "╣")
    tip = "💡 Lower threshold = better quality, less speedup"
    lines.append("║" + tip.center(width - 2) + "║")
    lines.append("╚" + "═" * (width - 2) + "╝")
    
    return "\n".join(lines)


def get_summary_stats(transformer: torch.nn.Module) -> Dict[str, Any]:
    """
    Get summary statistics from cache-dit.
    
    Returns dict with: total_steps, cached_steps, computed_steps,
    avg_residual_diff, max_residual_diff, speedup
    """
    try:
        import cache_dit
        
        stats = cache_dit.summary(transformer)
        
        if stats is None:
            return {}
        
        # Normalize stats to consistent format
        result = {
            "total_steps": getattr(stats, "total_steps", 0),
            "cached_steps": getattr(stats, "cached_steps", 0),
            "computed_steps": getattr(stats, "computed_steps", 0),
            "avg_residual_diff": getattr(stats, "avg_diff", 0.0),
            "max_residual_diff": getattr(stats, "max_diff", 0.0),
            "speedup": getattr(stats, "speedup", 1.0),
            "raw": stats,
        }
        
        # Calculate speedup if not provided
        if result["speedup"] == 1.0 and result["total_steps"] > 0:
            computed = result["computed_steps"] or (result["total_steps"] - result["cached_steps"])
            if computed > 0:
                result["speedup"] = result["total_steps"] / computed
        
        return result
        
    except Exception as e:
        logger.warning(f"Failed to get summary stats: {e}")
        return {}


def print_summary_to_log(
    transformer: torch.nn.Module,
    model_type: str,
    num_steps: int,
    config_info: Dict[str, Any],
) -> str:
    """
    Get summary, format as dashboard, and print to log.
    Returns the formatted string.
    """
    stats = get_summary_stats(transformer)
    dashboard = format_summary_dashboard(stats, model_type, num_steps, config_info)
    
    # Print to log
    logger.info("\n" + dashboard)
    print("\n" + dashboard)  # Also print to console
    
    return dashboard


# =============================================================================
# Noise Injection Utility
# =============================================================================

def apply_noise_injection(
    output: torch.Tensor,
    noise_scale: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Apply small noise perturbation to cached output.
    
    Prevents "static" or "dead" regions in generated content.
    Typical scale: 0.001 - 0.003
    """
    if noise_scale <= 0:
        return output
    
    noise = torch.randn_like(output, generator=generator) * noise_scale
    return output + noise
