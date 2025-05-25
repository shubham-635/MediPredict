import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelExplainer')


class ModelExplainer:
    """
    Simple model explainer without SHAP dependency
    """
    
    def __init__(self, model, feature_names=None, model_type='classifier', class_names=None):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.class_names = class_names
        
    def get_feature_importance(self, X_test, top_n=20):
        """Get feature importance from tree-based models"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names or [f'Feature_{i}' for i in range(len(self.model.feature_importances_))],
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False).head(top_n)
            
            return plt.gcf(), importance_df
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return None, None


class RiskVisualizer:
    """
    Class for creating risk visualizations for MediPredict.
    """
    
    def __init__(self, save_dir='visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_risk_gauge(self, risk_score, disease_name=None, save_path=None):
        """Create a simple risk gauge visualization"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Determine color based on risk level
        if risk_score >= 0.7:
            color = 'red'
            risk_level = 'High Risk'
        elif risk_score >= 0.4:
            color = 'orange'
            risk_level = 'Moderate Risk'
        else:
            color = 'green'
            risk_level = 'Low Risk'
        
        # Create a simple bar chart as gauge
        ax.barh(['Risk Level'], [risk_score], color=color)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Risk Score')
        
        # Add text
        ax.text(0.5, 0, f'{risk_score:.1%} - {risk_level}', 
                ha='center', va='bottom', transform=ax.transAxes, fontsize=16)
        
        if disease_name:
            ax.set_title(f'{disease_name} Risk Assessment', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Risk gauge saved to {save_path}")
            
        return fig
    
    def plot_risk_profile(self, risk_dict, patient_id=None, save_path=None):
        """Create a simple bar chart of disease risks"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        diseases = list(risk_dict.keys())
        risks = list(risk_dict.values())
        
        # Create color map based on risk levels
        colors = ['red' if r >= 0.7 else 'orange' if r >= 0.4 else 'green' for r in risks]
        
        bars = ax.bar(diseases, risks, color=colors)
        
        # Add value labels on bars
        for bar, risk in zip(bars, risks):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{risk:.1%}', ha='center', va='bottom')
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('Risk Score')
        ax.set_title(f'Disease Risk Profile{" - " + patient_id if patient_id else ""}')
        
        # Add risk level lines
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High Risk')
        ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Moderate Risk')
        
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Risk profile saved to {save_path}")
            
        return fig
    
    def plot_risk_factors(self, risk_factors, disease_name, max_factors=10, save_path=None):
        """Create a bar chart of risk factors"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract top factors
        factors = risk_factors[:max_factors]
        
        if not factors:
            ax.text(0.5, 0.5, 'No risk factors available', 
                    ha='center', va='center', transform=ax.transAxes)
        else:
            features = [f['feature'] for f in factors]
            importances = [f['importance'] for f in factors]
            
            ax.barh(features, importances)
            ax.set_xlabel('Importance')
            ax.set_title(f'Top Risk Factors for {disease_name}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Risk factors plot saved to {save_path}")
            
        return fig
