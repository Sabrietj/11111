"""
数据验证和质量检查工具
确保输入数据的一致性和有效性
"""

import numpy as np
import torch
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Union, List

logger = logging.getLogger(__name__)

class DataValidator:
    """数据验证和质量检查工具"""
    
    def __init__(self):
        self.validation_reports = []
    
    def ensure_tensor_consistency(self, data: Any) -> Union[np.ndarray, torch.Tensor]:
        """确保所有tensor数据转换为numpy/原生数值"""
        try:
            if torch.is_tensor(data):
                return data.detach().cpu().numpy()
            elif isinstance(data, np.ndarray):
                return data
            elif isinstance(data, (list, tuple)):
                try:
                    return np.array(data)
                except:
                    return data  # 如果无法转换，保持原样
            else:
                return data
        except Exception as e:
            logger.warning(f"数据类型转换失败: {e}")
            return data
    
    def data_quality_pipeline(self, feature_data: Any, feature_name: str, feature_type: str) -> Tuple[Dict[str, Any], Any]:
        """数据质量检查管道"""
        quality_report = {
            'feature_name': feature_name,
            'feature_type': feature_type,
            'valid': True,
            'issues': [],
            'warnings': [],
            'data_stats': {}
        }
        
        try:
            # 1. 检查数据完整性
            if feature_data is None:
                quality_report['valid'] = False
                quality_report['issues'].append("Data is None")
                return quality_report, feature_data
            
            # 2. 转换数据类型
            processed_data = self.ensure_tensor_consistency(feature_data)
            
            # 3. 数据形状检查
            if hasattr(processed_data, 'shape'):
                shape_info = {
                    'shape': list(processed_data.shape),
                    'ndim': len(processed_data.shape),
                    'size': processed_data.size if hasattr(processed_data, 'size') else 0
                }
                quality_report['data_stats']['shape_info'] = shape_info
                
                # 检查是否为标量
                if len(processed_data.shape) == 0:
                    quality_report['warnings'].append("Data is scalar")
                
                # 检查形状是否合理
                if len(processed_data.shape) > 3:
                    quality_report['warnings'].append(f"Data has high dimensionality: {processed_data.shape}")
            
            # 4. 数据类型检查
            if hasattr(processed_data, 'dtype'):
                dtype_info = {
                    'dtype': str(processed_data.dtype),
                    'is_floating': np.issubdtype(processed_data.dtype, np.floating) if hasattr(np, 'issubdtype') else False,
                    'is_integer': np.issubdtype(processed_data.dtype, np.integer) if hasattr(np, 'issubdtype') else False
                }
                quality_report['data_stats']['dtype_info'] = dtype_info
            
            # 5. 数值质量检查（针对数值数据）
            if isinstance(processed_data, (np.ndarray, torch.Tensor)) and len(processed_data.shape) > 0:
                # 转换为numpy进行统计计算
                if torch.is_tensor(processed_data):
                    np_data = processed_data.detach().cpu().numpy()
                else:
                    np_data = processed_data
                
                # 基本统计
                try:
                    stats = {
                        'count': np_data.size,
                        'has_nan': bool(np.isnan(np_data).any()),
                        'has_inf': bool(np.isinf(np_data).any()),
                        'zero_ratio': float((np_data == 0).mean()) if np_data.size > 0 else 0.0,
                        'min': float(np.min(np_data)) if np_data.size > 0 else 0.0,
                        'max': float(np.max(np_data)) if np_data.size > 0 else 0.0,
                        'mean': float(np.mean(np_data)) if np_data.size > 0 else 0.0,
                        'std': float(np.std(np_data)) if np_data.size > 0 else 0.0
                    }
                    quality_report['data_stats']['numerical_stats'] = stats
                    
                    # 处理异常值
                    if stats['has_nan']:
                        quality_report['warnings'].append("Contains NaN values")
                        if torch.is_tensor(processed_data):
                            processed_data = torch.nan_to_num(processed_data, nan=0.0)
                        else:
                            processed_data = np.nan_to_num(processed_data, nan=0.0)
                    
                    if stats['has_inf']:
                        quality_report['warnings'].append("Contains infinite values")
                        if torch.is_tensor(processed_data):
                            processed_data = torch.nan_to_num(processed_data, posinf=0.0, neginf=0.0)
                        else:
                            processed_data = np.nan_to_num(processed_data, posinf=0.0, neginf=0.0)
                    
                    # 特定类型的额外检查
                    if feature_type == 'sequence':
                        sequence_quality = self._check_sequence_quality(np_data, feature_name)
                        quality_report['data_stats']['sequence_quality'] = sequence_quality
                    
                    elif feature_type == 'text':
                        text_quality = self._check_text_data_quality(processed_data, feature_name)
                        quality_report['data_stats']['text_quality'] = text_quality
                    
                    elif feature_type == 'numeric':
                        numeric_quality = self._check_numeric_quality(np_data, feature_name)
                        quality_report['data_stats']['numeric_quality'] = numeric_quality
                    
                except Exception as e:
                    quality_report['warnings'].append(f"Statistical calculation failed: {e}")
            
            # 6. 特定特征类型的质量检查
            type_specific_issues = self._type_specific_validation(processed_data, feature_type, feature_name)
            quality_report['issues'].extend(type_specific_issues['issues'])
            quality_report['warnings'].extend(type_specific_issues['warnings'])
            
            # 7. 最终有效性检查
            if len(quality_report['issues']) > 0:
                quality_report['valid'] = False
            
            return quality_report, processed_data
            
        except Exception as e:
            quality_report['valid'] = False
            quality_report['issues'].append(f"Validation pipeline failed: {e}")
            logger.error(f"数据质量检查失败 for {feature_name}: {e}")
            return quality_report, feature_data
    
    def _check_sequence_quality(self, data: np.ndarray, feature_name: str) -> Dict[str, Any]:
        """检查序列数据质量"""
        quality = {
            'constant_sequences': 0,
            'zero_sequences': 0,
            'temporal_variance': 0.0,
            'sequence_diversity': 0.0
        }
        
        try:
            if len(data.shape) >= 2:
                # 检查每个序列的质量
                for i in range(min(data.shape[0], 10)):  # 最多检查10个序列
                    seq = data[i]
                    
                    # 检查是否为常数序列
                    if np.var(seq) < 1e-8:
                        quality['constant_sequences'] += 1
                    
                    # 检查是否为零序列
                    if np.all(seq == 0):
                        quality['zero_sequences'] += 1
                
                # 计算整体时序方差
                if len(data.shape) >= 2 and data.shape[1] > 1:
                    try:
                        diffs = np.diff(data, axis=1)
                        quality['temporal_variance'] = float(np.var(diffs))
                    except:
                        quality['temporal_variance'] = 0.0
                
                # 计算序列多样性
                try:
                    unique_sequences = len(set(tuple(seq) for seq in data[:10]))
                    quality['sequence_diversity'] = unique_sequences / min(len(data), 10)
                except:
                    quality['sequence_diversity'] = 0.0
        
        except Exception as e:
            logger.debug(f"序列质量检查失败: {e}")
        
        return quality
    
    def _check_text_data_quality(self, data: Any, feature_name: str) -> Dict[str, Any]:
        """检查文本数据质量"""
        quality = {
            'empty_text_ratio': 0.0,
            'avg_length': 0.0,
            'length_variance': 0.0,
            'unique_texts_ratio': 0.0
        }
        
        try:
            # 将数据转换为文本列表
            if isinstance(data, (list, tuple, np.ndarray)):
                texts = [str(item) for item in data if item is not None]
            else:
                texts = [str(data)]
            
            if texts:
                # 计算各种质量指标
                lengths = [len(text) for text in texts]
                empty_count = sum(1 for text in texts if not text.strip())
                
                quality['empty_text_ratio'] = empty_count / len(texts)
                quality['avg_length'] = np.mean(lengths)
                quality['length_variance'] = np.var(lengths)
                quality['unique_texts_ratio'] = len(set(texts)) / len(texts)
        
        except Exception as e:
            logger.debug(f"文本质量检查失败: {e}")
        
        return quality
    
    def _check_numeric_quality(self, data: np.ndarray, feature_name: str) -> Dict[str, Any]:
        """检查数值数据质量"""
        quality = {
            'skewness': 0.0,
            'kurtosis': 0.0,
            'outlier_ratio': 0.0,
            'value_range_ratio': 0.0
        }
        
        try:
            if data.size > 1:
                # 尝试导入scipy，如果失败则跳过偏度和峰度计算
                try:
                    from scipy import stats
                    
                    # 计算偏度和峰度
                    if hasattr(stats, 'skew') and hasattr(stats, 'kurtosis'):
                        quality['skewness'] = float(stats.skew(data.flatten()))
                        quality['kurtosis'] = float(stats.kurtosis(data.flatten()))
                except ImportError:
                    logger.debug("scipy not available, skipping skewness/kurtosis calculation")
                    quality['skewness'] = 0.0
                    quality['kurtosis'] = 0.0
                
                # 计算异常值比例
                q1, q3 = np.percentile(data.flatten(), [25, 75])
                iqr = q3 - q1
                if iqr > 1e-8:
                    outliers = ((data.flatten() < q1 - 1.5 * iqr) | 
                               (data.flatten() > q3 + 1.5 * iqr))
                    quality['outlier_ratio'] = float(np.mean(outliers))
                
                # 计算值范围比例
                data_range = np.max(data) - np.min(data)
                if data_range > 1e-8:
                    quality['value_range_ratio'] = data_range / (np.abs(np.mean(data)) + 1e-8)
        
        except Exception as e:
            logger.debug(f"数值质量检查失败: {e}")
        
        return quality
    
    def _type_specific_validation(self, data: Any, feature_type: str, feature_name: str) -> Dict[str, List[str]]:
        """特定类型的验证检查"""
        result = {'issues': [], 'warnings': []}
        
        try:
            if feature_type == 'sequence':
                if not hasattr(data, 'shape') or len(data.shape) < 2:
                    result['warnings'].append("Sequence data should be at least 2D")
                
                if hasattr(data, 'shape') and len(data.shape) >= 2 and data.shape[1] < 2:
                    result['warnings'].append("Sequence length is very short")
            
            elif feature_type == 'text':
                if isinstance(data, (list, np.ndarray)) and len(data) == 0:
                    result['issues'].append("Text data is empty")
            
            elif feature_type == 'numeric':
                if hasattr(data, 'dtype') and not np.issubdtype(data.dtype, np.number):
                    result['warnings'].append("Numeric feature has non-numeric dtype")
            
            elif feature_type == 'categorical':
                if hasattr(data, 'size') and data.size > 0:
                    unique_count = len(np.unique(data)) if isinstance(data, np.ndarray) else len(set(data))
                    if unique_count == 1:
                        result['warnings'].append("Categorical feature has only one category")
                    elif unique_count > data.size * 0.8:
                        result['warnings'].append("Categorical feature has too many unique values")
        
        except Exception as e:
            result['warnings'].append(f"Type-specific validation failed: {e}")
        
        return result
    
    def validate_batch(self, batch: Dict[str, Any], feature_classification: Dict[str, List[str]]) -> Dict[str, Any]:
        """验证整个批次的数据质量"""
        batch_report = {
            'overall_valid': True,
            'feature_reports': {},
            'summary': {
                'total_features': 0,
                'valid_features': 0,
                'features_with_issues': 0,
                'total_warnings': 0,
                'total_issues': 0
            }
        }
        
        try:
            for feature_type, feature_list in feature_classification.items():
                if feature_type == 'excluded' or not feature_list:
                    continue
                
                for feature_name in feature_list:
                    if feature_name in batch:
                        quality_report, processed_data = self.data_quality_pipeline(
                            batch[feature_name], feature_name, feature_type
                        )
                        
                        batch_report['feature_reports'][feature_name] = quality_report
                        batch_report['summary']['total_features'] += 1
                        
                        if quality_report['valid']:
                            batch_report['summary']['valid_features'] += 1
                        else:
                            batch_report['summary']['features_with_issues'] += 1
                            batch_report['overall_valid'] = False
                        
                        batch_report['summary']['total_issues'] += len(quality_report['issues'])
                        batch_report['summary']['total_warnings'] += len(quality_report['warnings'])
                        
                        # 更新原始批次数据（使用处理后的数据）
                        batch[feature_name] = processed_data
            
            logger.info(f"批次数据质量检查完成: {batch_report['summary']['valid_features']}/{batch_report['summary']['total_features']} 特征有效")
            return batch_report
            
        except Exception as e:
            logger.error(f"批次数据验证失败: {e}")
            batch_report['overall_valid'] = False
            batch_report['summary']['validation_error'] = str(e)
            return batch_report
    
    def get_validation_summary(self, validation_report: Dict[str, Any]) -> str:
        """生成验证摘要文本"""
        try:
            summary = validation_report.get('summary', {})
            
            text = f"""=== 数据质量验证摘要 ===
总特征数: {summary.get('total_features', 0)}
有效特征数: {summary.get('valid_features', 0)}
有问题特征数: {summary.get('features_with_issues', 0)}
总警告数: {summary.get('total_warnings', 0)}
总问题数: {summary.get('total_issues', 0)}
整体状态: {'✓ 通过' if validation_report.get('overall_valid', False) else '✗ 失败'}
"""
            
            # 添加有问题的特征详情
            for feature_name, report in validation_report.get('feature_reports', {}).items():
                if not report.get('valid', True):
                    text += f"\n❌ {feature_name}:\n"
                    for issue in report.get('issues', []):
                        text += f"  - 问题: {issue}\n"
                    for warning in report.get('warnings', []):
                        text += f"  - 警告: {warning}\n"
            
            return text
            
        except Exception as e:
            return f"生成验证摘要失败: {e}"