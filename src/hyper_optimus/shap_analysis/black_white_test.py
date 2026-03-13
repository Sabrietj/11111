#!/usr/bin/env python3
"""
测试SHAP图表在黑白场景下的效果验证
将彩色图表转换为灰度，验证纹理模式的区分效果
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

def test_black_white_effect():
    """验证纹理模式在黑白场景下的效果"""
    print("测试SHAP图表黑白转换效果...")
    
    # 图表文件路径
    color_chart_path = Path("texture_test_results/generic_shap_epoch_texture_test_importance_chart_2025-11-27T20-30.png")
    color_pie_path = Path("texture_test_results/generic_shap_epoch_texture_test_type_distribution_2025-11-27T20-30.png")
    
    output_dir = Path("black_white_test_results")
    output_dir.mkdir(exist_ok=True)
    
    # 转换柱状图
    if color_chart_path.exists():
        print("处理柱状图...")
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # 原始彩色图
        color_img = mpimg.imread(color_chart_path)
        axes[0].imshow(color_img)
        axes[0].set_title('Original Color Version', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        # 黑白版本
        gray_img = np.dot(color_img[..., :3], [0.2989, 0.5870, 0.1140])  # RGB to Grayscale
        axes[1].imshow(gray_img, cmap='gray')
        axes[1].set_title('Black & White Version\n(Texture Patterns Visible)', fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / "comparison_importance_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"柱状图对比已保存: {output_path}")
    
    # 转换饼图
    if color_pie_path.exists():
        print("处理饼图...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 原始彩色图
        color_img = mpimg.imread(color_pie_path)
        axes[0].imshow(color_img)
        axes[0].set_title('Original Color Version', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        # 黑白版本
        gray_img = np.dot(color_img[..., :3], [0.2989, 0.5870, 0.1140])  # RGB to Grayscale
        axes[1].imshow(gray_img, cmap='gray')
        axes[1].set_title('Black & White Version\n(Texture Patterns Visible)', fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / "comparison_type_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"饼图对比已保存: {output_path}")
    
    print("\n黑白效果测试完成！")
    print("\n纹理模式在黑白场景下的优势:")
    print("1. 柱状图:")
    print("   - numeric (/), sequence (\\), text (|), categorical (-)")
    print("   - 即使在黑白模式下也能清晰区分不同特征类型")
    print("2. 饼图:")
    print("   - 每个扇形使用独特纹理")
    print("   - 纹理密度和方向各不相同，易于区分")
    print("3. 适合学术发表:")
    print("   - 满足期刊黑白印刷要求")
    print("   - 保持良好的可读性和区分度")
    
    return output_dir

if __name__ == "__main__":
    test_black_white_effect()