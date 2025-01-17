import time
import os
import argparse
import copy
import numpy as np
import cv2
import pandas as pd
import json
import torch
from tifffile import imread
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from pathlib import Path
import jinja2
import webbrowser
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QComboBox, 
                           QFileDialog, QProgressBar, QMessageBox, QCheckBox, 
                           QFrame, QMenuBar, QMenu, QDialog, QFormLayout,
                           QLineEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QAction, QActionGroup
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class LanguageManager:
    def __init__(self):
        self.current_language = 'en'
        self.translations = {}
        self.load_translations()
        
    def load_translations(self):
        i18n_dir = os.path.join(os.path.dirname(__file__), 'i18n')
        for lang_file in os.listdir(i18n_dir):
            if lang_file.endswith('.json'):
                lang_code = os.path.splitext(lang_file)[0]
                with open(os.path.join(i18n_dir, lang_file), 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
    
    def set_language(self, lang_code):
        if lang_code in self.translations:
            self.current_language = lang_code
            return True
        return False
    
    def get_text(self, key_path):
        """Get translated text using dot notation for nested keys"""
        try:
            current_dict = self.translations[self.current_language]
            for key in key_path.split('.'):
                current_dict = current_dict[key]
            return current_dict
        except (KeyError, TypeError):
            return key_path

# Global instances
language_manager = LanguageManager()

# Global variables
MODEL_CONFIG = {
    'checkpoint': "checkpoints/sam2.1_hiera_large.pt",
    'config': get_config_path("configs/sam2.1/sam2.1_hiera_l.yaml")
}

MODEL_SIZES = {
    'tiny': {
        'checkpoint': "sam2.1_hiera_tiny.pt",
        'config': "configs/sam2.1/sam2.1_hiera_t.yaml",
        'model_config': {
            'embed_dim': 384,
            'num_heads': 6,
            'depths': [2, 2, 6, 2],
            'window_size': 7,
            'mlp_ratio': 4
        }
    },
    'small': {
        'checkpoint': "sam2.1_hiera_small.pt", 
        'config': "configs/sam2.1/sam2.1_hiera_s.yaml",
        'model_config': {
            'embed_dim': 384,
            'num_heads': 6,
            'depths': [2, 2, 18, 2],
            'window_size': 7,
            'mlp_ratio': 4
        }
    },
    'base': {
        'checkpoint': "sam2.1_hiera_base_plus.pt",
        'config': "configs/sam2.1/sam2.1_hiera_b+.yaml",
        'model_config': {
            'embed_dim': 512,
            'num_heads': 8,
            'depths': [2, 2, 18, 2],
            'window_size': 7,
            'mlp_ratio': 4
        }
    },
    'large': {
        'checkpoint': "sam2.1_hiera_large.pt",
        'config': "configs/sam2.1/sam2.1_hiera_l.yaml",
        'model_config': {
            'embed_dim': 512,
            'num_heads': 8,
            'depths': [2, 2, 18, 2],
            'window_size': 7,
            'mlp_ratio': 4
        }
    }
}

def get_config_path(config_file):
    """Get the absolute path for config file"""
    # Try to find config in sam2 package first
    sam2_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_paths = [
        os.path.join(sam2_path, config_file),  # Try sam2 package path
        os.path.join(os.path.dirname(__file__), config_file),  # Try current directory
        config_file  # Try absolute path
    ]
    
    for path in config_paths:
        if os.path.exists(path):
            return path
    return config_file  # Return original path if not found

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings('ISCO', 'ISCO')
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(language_manager.get_text('settings_dialog.title'))
        layout = QFormLayout()
        
        # Checkpoints directory
        self.checkpoint_edit = QLineEdit(self)
        self.checkpoint_edit.setText(self.settings.value('checkpoint_dir', 'checkpoints'))
        browse_btn = QPushButton(language_manager.get_text('main_window.browse'), self)
        browse_btn.clicked.connect(self.browse_checkpoint_dir)
        
        checkpoint_layout = QHBoxLayout()
        checkpoint_layout.addWidget(self.checkpoint_edit)
        checkpoint_layout.addWidget(browse_btn)
        
        layout.addRow(language_manager.get_text('settings_dialog.checkpoints_dir'), checkpoint_layout)
        
        # Save button
        save_btn = QPushButton(language_manager.get_text('settings_dialog.save'), self)
        save_btn.clicked.connect(self.save_settings)
        layout.addRow(save_btn)
        
        self.setLayout(layout)
        
    def browse_checkpoint_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Checkpoints Directory")
        if dir_path:
            self.checkpoint_edit.setText(dir_path)
            
    def save_settings(self):
        checkpoint_dir = self.checkpoint_edit.text()
        self.settings.setValue('checkpoint_dir', checkpoint_dir)
        
        # Update MODEL_CONFIG with new paths
        global MODEL_CONFIG
        MODEL_CONFIG = {
            'checkpoint': os.path.join(checkpoint_dir, os.path.basename(MODEL_CONFIG['checkpoint'])),
            'config': os.path.join(checkpoint_dir, os.path.basename(MODEL_CONFIG['config']))
        }
        
        self.accept()

class ProcessThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, input_dir, output_dir, map_file=None, crop_size=10000):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.map_file = map_file
        self.crop_size = crop_size
        self.is_running = True
        
    def stop(self):
        self.is_running = False
        
    def run(self):
        try:
            # Load map file if provided
            group_map = None
            if self.map_file and os.path.exists(self.map_file):
                try:
                    map_df = pd.read_csv(self.map_file)
                    group_map = dict(zip(map_df['Source'], map_df['Group']))
                    self.status.emit("Map file loaded successfully")
                except Exception as e:
                    self.status.emit(f"Warning: Could not load map file: {str(e)}")
            
            # Initialize device and model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.status.emit("Initializing model...")
            sam2 = build_sam2(MODEL_CONFIG['config'], MODEL_CONFIG['checkpoint'], 
                            device=device, apply_postprocessing=False)
            mask_generator = SAM2AutomaticMaskGenerator(sam2)
            
            # Get list of PNG files
            png_files = [f for f in os.listdir(self.input_dir) if f.endswith('.png')]
            total_files = len(png_files)
            
            for idx, file in enumerate(png_files, 1):
                if not self.is_running:
                    self.status.emit("Processing aborted")
                    return
                    
                self.status.emit(f"Processing {file}...")
                fn = os.path.join(self.input_dir, file)
                cell_properties = {}
                
                try:
                    image = cv2.imread(fn)
                except:
                    image = imread(fn)
                
                cell_properties_patch, sta_idx, cell_mask_batch = patch_property(
                    image, 0, mask_generator)
                cell_properties.update(cell_properties_patch)
                
                cv2.imwrite(os.path.join(self.output_dir, 
                           file.strip('.png') + '_cell_mask.png'),
                           cell_mask_batch)
                
                # Convert to DataFrame and add group information if map exists
                cell_properties = pd.DataFrame.from_dict(cell_properties, orient='index')
                if group_map is not None:
                    # Extract the base name from the file (without extension)
                    base_name = os.path.splitext(file)[0]
                    if base_name in group_map:
                        cell_properties['Group'] = group_map[base_name]
                
                cell_properties.to_csv(os.path.join(self.output_dir, 
                                     file.replace('.png', '.csv')))
                
                self.progress.emit(int(idx / total_files * 100))
                
            if self.is_running:
                self.status.emit("Processing completed!")
                self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))

class ReportGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.report_dir = os.path.join(output_dir, 'report')
        os.makedirs(self.report_dir, exist_ok=True)
        
    def generate_plots(self, df, group_col='Group'):
        """Generate visualization plots for the data"""
        plot_dir = os.path.join(self.report_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        metrics = ['area', 'perimeter', 'radius', 'non-smoothness', 'non-circularity', 'symmetry']
        plot_paths = {}
        
        # Set style
        plt.style.use('seaborn')
        
        # Distribution plots for each metric
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x=group_col, y=metric)
            plt.title(f'{metric.replace("_", " ").title()} Distribution by Group')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_path = os.path.join(plot_dir, f'{metric}_boxplot.png')
            plt.savefig(plot_path)
            plt.close()
            plot_paths[f'{metric}_boxplot'] = plot_path
            
            # Violin plots for detailed distribution
            plt.figure(figsize=(10, 6))
            sns.violinplot(data=df, x=group_col, y=metric)
            plt.title(f'{metric.replace("_", " ").title()} Distribution (Violin) by Group')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_path = os.path.join(plot_dir, f'{metric}_violin.png')
            plt.savefig(plot_path)
            plt.close()
            plot_paths[f'{metric}_violin'] = plot_path
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[metrics].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Cell Properties')
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, 'correlation_heatmap.png')
        plt.savefig(plot_path)
        plt.close()
        plot_paths['correlation_heatmap'] = plot_path
        
        return plot_paths
    
    def perform_statistical_analysis(self, df, group_col='Group'):
        """Perform statistical analysis on the data"""
        metrics = ['area', 'perimeter', 'radius', 'non-smoothness', 'non-circularity', 'symmetry']
        results = {}
        
        # Descriptive statistics by group
        desc_stats = df.groupby(group_col)[metrics].describe()
        results['descriptive'] = desc_stats
        
        # ANOVA and Tukey's HSD test for group comparisons
        anova_results = {}
        tukey_results = {}
        groups = df[group_col].unique()
        
        for metric in metrics:
            # One-way ANOVA
            groups_data = [df[df[group_col] == group][metric] for group in groups]
            f_stat, p_val = stats.f_oneway(*groups_data)
            anova_results[metric] = {'f_statistic': f_stat, 'p_value': p_val}
            
            # Tukey's HSD test for pairwise comparisons
            if len(groups) > 2:
                tukey = stats.tukey_hsd(*groups_data)
                tukey_results[metric] = {
                    'groups': list(groups),
                    'statistics': tukey
                }
        
        results['anova'] = anova_results
        results['tukey'] = tukey_results
        
        return results
    
    def generate_html_report(self, df, plot_paths, stats_results):
        """Generate HTML report with analysis results"""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ISCO Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2, h3 { color: #2c3e50; }
                .header { text-align: center; margin-bottom: 40px; }
                .subtitle { color: #7f8c8d; font-size: 1.2em; margin-bottom: 20px; }
                .plot-container { margin: 20px 0; }
                .plot-container img { max-width: 100%; height: auto; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f5f5f5; }
                .significant { color: #e74c3c; }
                .section { margin: 30px 0; }
                .footer { margin-top: 40px; text-align: center; color: #7f8c8d; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ISCO Analysis Report</h1>
                <div class="subtitle">Intelligent Framework for Accurate Segmentation and Comparative Analysis of Organoids</div>
                <p>Generated on: {{ datetime }}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>This report presents a comprehensive analysis of organoid properties across different groups, 
                including morphological characteristics and statistical comparisons.</p>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                {{ descriptive_stats | safe }}
            </div>
            
            <div class="section">
                <h2>Morphological Analysis</h2>
                <h3>Distribution Plots</h3>
                {% for metric in metrics %}
                <div class="plot-container">
                    <h4>{{ metric | title }}</h4>
                    <img src="{{ plot_paths[metric + '_boxplot'] }}" alt="{{ metric }} boxplot">
                    <img src="{{ plot_paths[metric + '_violin'] }}" alt="{{ metric }} violin plot">
                </div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h3>Property Correlations</h3>
                <div class="plot-container">
                    <img src="{{ plot_paths['correlation_heatmap'] }}" alt="Correlation heatmap">
                </div>
            </div>
            
            <div class="section">
                <h2>Statistical Analysis</h2>
                <h3>One-way ANOVA Results</h3>
                {{ anova_table | safe }}
                
                {% if tukey_results %}
                <h3>Post-hoc Analysis (Tukey's HSD)</h3>
                {{ tukey_table | safe }}
                {% endif %}
            </div>
            
            <div class="footer">
                <p>ISCO - Advanced Organoid Analysis Platform</p>
                <p>Report generated using automated analysis pipeline</p>
            </div>
        </body>
        </html>
        """
        
        # Prepare template data
        template_data = {
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'descriptive_stats': stats_results['descriptive'].to_html(),
            'metrics': ['area', 'perimeter', 'radius', 'non-smoothness', 'non-circularity', 'symmetry'],
            'plot_paths': {k: os.path.relpath(v, self.report_dir) for k, v in plot_paths.items()},
            'anova_table': pd.DataFrame(stats_results['anova']).T.to_html(),
            'tukey_results': None
        }
        
        # Generate HTML
        template = jinja2.Template(template_str)
        html_content = template.render(**template_data)
        
        # Save report
        report_path = os.path.join(self.report_dir, 'analysis_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path

class CitationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(language_manager.get_text('menu.about'))
        self.setMinimumWidth(600)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Software title and description
        title_label = QLabel("ISCO")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title_label)
        
        desc_label = QLabel(language_manager.get_text('main_window.description'))
        desc_label.setStyleSheet("font-size: 12px; color: #7f8c8d;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Version info
        version_label = QLabel("Version: 1.0.0")
        layout.addWidget(version_label)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # Citation section
        citation_label = QLabel("Citation:")
        citation_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(citation_label)
        
        citation_text = QLabel(
            "When using ISCO in your research, please cite the following paper:\n\n"
            "Zhou J, Fu Z, Ni X, et al. ISCO: Intelligent Framework for Accurate "
            "Segmentation and Comparative Analysis of Organoids[J]. bioRxiv, "
            "2024: 2024.12.24.630244."
        )
        citation_text.setWordWrap(True)
        citation_text.setStyleSheet("background-color: #f8f9fa; padding: 10px; border-radius: 5px;")
        layout.addWidget(citation_text)
        
        # Copy button
        copy_btn = QPushButton("Copy Citation")
        copy_btn.clicked.connect(lambda: self.copy_to_clipboard(
            "Zhou J, Fu Z, Ni X, et al. ISCO: Intelligent Framework for Accurate "
            "Segmentation and Comparative Analysis of Organoids[J]. bioRxiv, "
            "2024: 2024.12.24.630244."
        ))
        copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #2c3e50;
                color: white;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
        """)
        layout.addWidget(copy_btn)
        
        # Copyright
        copyright_label = QLabel("© 2024 ISCO Team")
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(copyright_label)
        
        self.setLayout(layout)
    
    def copy_to_clipboard(self, text):
        """Copy text to clipboard and show feedback"""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        QMessageBox.information(self, "Success", "Citation copied to clipboard!")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings('ISCO', 'ISCO')
        self.ensure_resource_directories()
        self.check_model_files()
        self.load_language_setting()
        self.load_model_config()
        self.updateUI()
        
    def ensure_resource_directories(self):
        """Ensure all required resource directories exist"""
        checkpoint_dir = self.settings.value('checkpoint_dir', 'checkpoints')
        
        # Create directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create assets directory if it doesn't exist
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        os.makedirs(assets_dir, exist_ok=True)

    def check_model_files(self):
        """Check if model files exist and show guidance if they don't"""
        checkpoint_dir = self.settings.value('checkpoint_dir', 'checkpoints')
        current_size = self.settings.value('model_size', 'large')
        
        if current_size in MODEL_SIZES:
            checkpoint_path = os.path.join(checkpoint_dir, MODEL_SIZES[current_size]['checkpoint'])
            config_path = get_config_path(MODEL_SIZES[current_size]['config'])
            
            if not os.path.exists(checkpoint_path):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setWindowTitle(language_manager.get_text('messages.model_not_found'))
                msg.setText(language_manager.get_text('messages.download_model'))
                msg.setInformativeText(
                    f"Please download the model checkpoint and place it in the following location:\n\n"
                    f"Model checkpoint:\n   {checkpoint_path}\n\n"
                    f"You can download the model files from our official repository."
                )
                msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg.exec()
    
    def load_language_setting(self):
        lang = self.settings.value('language', 'en')
        language_manager.set_language(lang)
        
    def load_model_config(self):
        """Load model configuration based on settings"""
        checkpoint_dir = self.settings.value('checkpoint_dir', 'checkpoints')
        current_size = self.settings.value('model_size', 'large')
        
        global MODEL_CONFIG
        if current_size in MODEL_SIZES:
            MODEL_CONFIG = {
                'checkpoint': os.path.join(checkpoint_dir, MODEL_SIZES[current_size]['checkpoint']),
                'config': get_config_path(MODEL_SIZES[current_size]['config']),
                'model_config': MODEL_SIZES[current_size]['model_config']
            }
    
    def createMenuBar(self):
        menubar = self.menuBar()
        
        # Language Menu
        language_menu = menubar.addMenu(language_manager.get_text('menu.language'))
        english_action = QAction('English', self)
        chinese_action = QAction('中文', self)
        english_action.triggered.connect(lambda: self.change_language('en'))
        chinese_action.triggered.connect(lambda: self.change_language('zh'))
        language_menu.addAction(english_action)
        language_menu.addAction(chinese_action)
        
        # Model Settings Menu - using different text based on language
        model_menu_text = ('Settings' if language_manager.current_language == 'en' 
                          else '模型选择')
        model_menu = menubar.addMenu(model_menu_text)
        model_group = QActionGroup(self)
        model_group.setExclusive(True)
        
        # Get current model size from settings
        current_size = self.settings.value('model_size', 'large')
        
        for size in MODEL_SIZES.keys():
            action = QAction(size.capitalize(), self, checkable=True)
            action.setChecked(size == current_size)
            action.triggered.connect(lambda checked, s=size: self.change_model_size(s))
            model_group.addAction(action)
            model_menu.addAction(action)
        
        # Guide Menu
        guide_menu = menubar.addMenu(language_manager.get_text('menu.guide'))
        user_guide_action = QAction(language_manager.get_text('menu.user_guide'), self)
        user_guide_action.triggered.connect(self.show_user_guide)
        about_action = QAction(language_manager.get_text('menu.about'), self)
        about_action.triggered.connect(self.show_about)
        guide_menu.addAction(user_guide_action)
        guide_menu.addAction(about_action)
    
    def change_language(self, lang_code):
        if language_manager.set_language(lang_code):
            self.settings.setValue('language', lang_code)
            # Recreate UI with new language
            self.close()
            self.__init__()
            self.show()
    
    def show_user_guide(self):
        """Show user guide in default browser"""
        guide_path = os.path.join(os.path.dirname(__file__), 'assets', 'user_guide.html')
        if os.path.exists(guide_path):
            webbrowser.open(f'file://{os.path.abspath(guide_path)}')
        else:
            QMessageBox.warning(self, 
                              language_manager.get_text('messages.error'),
                              language_manager.get_text('messages.guide_not_found'))
    
    def show_about(self):
        """Show about dialog with citation information"""
        dialog = CitationDialog(self)
        dialog.exec()
    
    def initUI(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add title and description
        title_label = QLabel(language_manager.get_text('main_window.title'))
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title_label)
        
        desc_label = QLabel(language_manager.get_text('main_window.description'))
        desc_label.setStyleSheet("font-size: 12px; color: #7f8c8d; margin-bottom: 10px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Add separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # Input directory selection
        input_layout = QHBoxLayout()
        self.input_label = QLabel(language_manager.get_text('main_window.input_dir'))
        self.input_path = QLabel(language_manager.get_text('main_window.not_selected'))
        self.input_btn = QPushButton(language_manager.get_text('main_window.browse'))
        self.input_btn.clicked.connect(self.select_input_dir)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(self.input_btn)
        layout.addLayout(input_layout)
        
        # Output directory selection
        output_layout = QHBoxLayout()
        self.output_label = QLabel(language_manager.get_text('main_window.output_dir'))
        self.output_path = QLabel(language_manager.get_text('main_window.not_selected'))
        self.output_btn = QPushButton(language_manager.get_text('main_window.browse'))
        self.output_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(self.output_btn)
        layout.addLayout(output_layout)
        
        # Map file selection
        map_layout = QHBoxLayout()
        self.map_label = QLabel(language_manager.get_text('main_window.map_file'))
        self.map_path = QLabel(language_manager.get_text('main_window.not_selected'))
        self.map_btn = QPushButton(language_manager.get_text('main_window.browse'))
        self.map_btn.clicked.connect(self.select_map_file)
        map_layout.addWidget(self.map_label)
        map_layout.addWidget(self.map_path)
        map_layout.addWidget(self.map_btn)
        layout.addLayout(map_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel(language_manager.get_text('main_window.ready'))
        layout.addWidget(self.status_label)
        
        # Add horizontal layout for buttons
        button_layout = QHBoxLayout()
        
        # Process button
        self.process_btn = QPushButton(language_manager.get_text('main_window.start_processing'))
        self.process_btn.clicked.connect(self.start_processing)
        button_layout.addWidget(self.process_btn)
        
        # Abort button
        self.abort_btn = QPushButton(language_manager.get_text('main_window.abort_processing'))
        self.abort_btn.clicked.connect(self.abort_processing)
        self.abort_btn.setEnabled(False)
        button_layout.addWidget(self.abort_btn)
        
        # Merge data button
        self.merge_btn = QPushButton(language_manager.get_text('main_window.merge_data'))
        self.merge_btn.clicked.connect(self.merge_data)
        self.merge_btn.setEnabled(False)
        button_layout.addWidget(self.merge_btn)
        
        layout.addLayout(button_layout)
        
        # Initialize processing thread as None
        self.process_thread = None
        
    def select_input_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if dir_path:
            self.input_path.setText(dir_path)
            
    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_path.setText(dir_path)
            
    def select_map_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Map File", "", "CSV Files (*.csv);;All Files (*.*)")
        if file_path:
            self.map_path.setText(file_path)
            
    def start_processing(self):
        if self.input_path.text() == language_manager.get_text('main_window.not_selected'):
            QMessageBox.warning(self, 
                              language_manager.get_text('messages.error'),
                              language_manager.get_text('messages.select_input'))
            return
        if self.output_path.text() == language_manager.get_text('main_window.not_selected'):
            QMessageBox.warning(self, 
                              language_manager.get_text('messages.error'),
                              language_manager.get_text('messages.select_output'))
            return
            
        self.process_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        map_file = None if self.map_path.text() == language_manager.get_text('main_window.not_selected') else self.map_path.text()
        
        self.process_thread = ProcessThread(
            self.input_path.text(),
            self.output_path.text(),
            map_file
        )
        
        self.process_thread.progress.connect(self.update_progress)
        self.process_thread.status.connect(self.update_status)
        self.process_thread.finished.connect(self.processing_finished)
        self.process_thread.error.connect(self.processing_error)
        
        self.process_thread.start()
        
    def abort_processing(self):
        if self.process_thread and self.process_thread.isRunning():
            self.process_thread.stop()
            self.process_thread.wait()
            self.process_btn.setEnabled(True)
            self.abort_btn.setEnabled(False)
            self.status_label.setText(language_manager.get_text('messages.processing_aborted'))
            
    def processing_finished(self):
        self.process_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        self.merge_btn.setEnabled(True)
        QMessageBox.information(self, 
                              language_manager.get_text('messages.success'),
                              language_manager.get_text('messages.processing_complete'))
        
    def processing_error(self, error_message):
        self.process_btn.setEnabled(True)
        QMessageBox.critical(self, 
                           language_manager.get_text('messages.error'),
                           f"{language_manager.get_text('messages.error')}: {error_message}")
        
    def merge_data(self):
        if self.output_path.text() == language_manager.get_text('main_window.not_selected'):
            QMessageBox.warning(self, 
                              language_manager.get_text('messages.error'),
                              language_manager.get_text('messages.no_output'))
            return
            
        try:
            # Collect all CSV files in the output directory
            output_dir = self.output_path.text()
            csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
            
            if not csv_files:
                QMessageBox.warning(self, 
                                  language_manager.get_text('messages.error'),
                                  language_manager.get_text('messages.no_csv'))
                return
            
            # Load map file if provided
            group_map = None
            if self.map_path.text() != language_manager.get_text('main_window.not_selected'):
                try:
                    map_df = pd.read_csv(self.map_path.text())
                    group_map = dict(zip(map_df['Source'], map_df['Group']))
                except Exception as e:
                    QMessageBox.warning(self, 
                                      language_manager.get_text('messages.error'),
                                      language_manager.get_text('messages.no_map'))
            
            # Combine all CSV files
            dfs = []
            for csv_file in csv_files:
                df = pd.read_csv(os.path.join(output_dir, csv_file))
                base_name = os.path.splitext(csv_file)[0]
                
                # Add source and group information
                df['Source'] = base_name
                if group_map and base_name in group_map:
                    df['Group'] = group_map[base_name]
                
                dfs.append(df)
            
            # Combine all dataframes
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Save merged data
            merged_file = os.path.join(output_dir, 'merged_data.csv')
            combined_df.to_csv(merged_file, index=False)
            
            QMessageBox.information(self, 
                                  language_manager.get_text('messages.success'),
                                  language_manager.get_text('messages.data_merged'))
            
        except Exception as e:
            QMessageBox.critical(self, 
                               language_manager.get_text('messages.error'),
                               f"{language_manager.get_text('messages.error')}: {str(e)}")

    def updateUI(self):
        """Update the UI with current language settings"""
        self.setWindowTitle(language_manager.get_text('window_title'))
        self.createMenuBar()
        self.initUI()

    def update_progress(self, progress):
        """Update the progress bar with the given progress value"""
        self.progress_bar.setValue(progress)
        QApplication.processEvents()

    def update_status(self, message):
        """Update the status label with the given message."""
        if hasattr(self, 'status_label'):
            self.status_label.setText(message)
            QApplication.processEvents()

    def change_model_size(self, size):
        """Change the model size and update settings"""
        if size in MODEL_SIZES:
            self.settings.setValue('model_size', size)
            checkpoint_dir = self.settings.value('checkpoint_dir', 'checkpoints')
            
            # Update MODEL_CONFIG with new paths and configuration
            global MODEL_CONFIG
            MODEL_CONFIG = {
                'checkpoint': os.path.join(checkpoint_dir, MODEL_SIZES[size]['checkpoint']),
                'config': get_config_path(MODEL_SIZES[size]['config']),
                'model_config': MODEL_SIZES[size]['model_config']
            }
            
            # Verify that the model files exist
            if not os.path.exists(MODEL_CONFIG['checkpoint']):
                QMessageBox.warning(
                    self,
                    language_manager.get_text('messages.error'),
                    f"Model checkpoint not found: {MODEL_CONFIG['checkpoint']}"
                )
                return
            
            # Update the model configuration
            try:
                # If a model is currently loaded, you might want to reinitialize it
                if hasattr(self, 'process_thread') and self.process_thread is not None:
                    self.process_thread.wait()  # Wait for any ongoing processing
                    
                QMessageBox.information(
                    self,
                    language_manager.get_text('messages.success'),
                    language_manager.get_text('messages.model_changed').format(size=size)
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    language_manager.get_text('messages.error'),
                    f"Error updating model configuration: {str(e)}"
                )

# Utility Functions
def remove_edge_cells(mask_image):
    w, h = mask_image.shape
    pruned_mask = copy.deepcopy(mask_image)
    remove_list = []
    edges = mask_image[0,:], mask_image[w-1,:], mask_image[:,0], mask_image[:,h-1]
    for edge in edges:
        edge_masks = np.unique(edge)
        for edge_mask in edge_masks:
            remove_list.append(edge_mask)
            pruned_mask[np.where(mask_image==edge_mask)] = 0
    return pruned_mask

def remove_small_cells(mask_image, area_threshold=10000):
    pruned_mask = copy.deepcopy(mask_image)
    for mask_index in np.unique(mask_image):
        area = np.sum(mask_image == mask_index)
        if area < area_threshold:
            pruned_mask[np.where(mask_image == mask_index)] = 0
    return pruned_mask

def split_cell_masks(mask_image):
    gray_mask = cv2.cvtColor(np.array(mask_image*255, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    unique_colors = np.unique(mask_image.reshape(-1, mask_image.shape[2]), axis=0)
    index_mask = np.zeros_like(gray_mask)
    
    for i, color in enumerate(unique_colors):
        color_mask = np.all(mask_image == color, axis=2)
        index_mask[color_mask] = i + 1
    
    index_mask[np.where(gray_mask==255)] = 0
    return index_mask

def remove_concentric_masks(mask_image):
    cell_values = np.unique(mask_image)
    for i in range(1, len(cell_values)):
        mask_one = np.array(mask_image == cell_values[i], dtype=np.uint8)
        contour, _ = cv2.findContours(mask_one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contour) > 0:
            largest_contour = max(contour, key=cv2.contourArea)
            mask_image = cv2.drawContours(mask_image, [largest_contour], -1, int(cell_values[i]), thickness=cv2.FILLED)
    return mask_image

def analyze_cell_properties(mask):
    properties = {}
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_idx = np.argmax([len(contour[i]) for i in range(len(contour))])
    
    perimeter = cv2.arcLength(contour[contour_idx], True)
    area = cv2.contourArea(contour[contour_idx])
    _, radius = cv2.minEnclosingCircle(contour[contour_idx])
    
    ellipse = cv2.fitEllipse(contour[contour_idx])
    ellipse_contour = cv2.ellipse2Poly(
        (int(ellipse[0][0]), int(ellipse[0][1])),
        (int(ellipse[1][0] * 0.5), int(ellipse[1][1] * 0.5)),
        int(ellipse[2]), 0, 360, 5
    )
    
    perimeter_ellipse = cv2.arcLength(ellipse_contour, closed=True)
    smoothness = perimeter_ellipse / perimeter
    compactness = abs((perimeter ** 2) / (area * 4 * np.pi) - 1)
    symmetry = cv2.matchShapes(contour[contour_idx], cv2.convexHull(contour[contour_idx], returnPoints=True), 1, 0.0)
    
    properties.update({
        'perimeter': perimeter,
        'area': area,
        'radius': radius,
        'non-smoothness': smoothness,
        'non-circularity': compactness,
        'symmetry': symmetry
    })
    return properties

# Analysis Functions
def combine_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    
    for ann in sorted_anns:
        m = ann['segmentation']
        xmin, ymin, xmax, ymax = ann['bbox']
        
        if abs(xmin-0) < 5 and abs(ymin-0) < 5 and \
           abs(xmax-sorted_anns[0]['segmentation'].shape[0]) < 5 and \
           abs(ymax-sorted_anns[0]['segmentation'].shape[1]) < 5:
            pass
        else:
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
    return img

def patch_property(image, start_idx, mask_generator):
    masks = mask_generator.generate(image)
    final_mask = combine_anns(masks)
    index_mask = split_cell_masks(final_mask)
    
    pruned_mask = remove_edge_cells(index_mask)
    pruned_mask_reduce = remove_small_cells(pruned_mask, area_threshold=1500)
    pruned_mask_reduce = remove_concentric_masks(pruned_mask_reduce)
    cell_mask = np.zeros((pruned_mask.shape[0], pruned_mask.shape[1], 3))
    
    cell_num = len(np.unique(pruned_mask_reduce)) - 1
    properties = {}
    
    for i in range(1, cell_num+1):
        mask_one = np.array(pruned_mask_reduce == np.unique(pruned_mask_reduce)[i], dtype=np.uint8)
        try:
            properties[f'cell {i+start_idx}'] = analyze_cell_properties(mask_one)
            cell_color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            contours, _ = cv2.findContours(mask_one, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(cell_mask, contours, -1, cell_color, 3)
            
            text = str(i+start_idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            
            text_x = (np.max(np.where(mask_one==1)[0]) - np.min(np.where(mask_one==1)[0]))//2 + np.min(np.where(mask_one==1)[0])
            text_y = (np.max(np.where(mask_one==1)[1]) - np.min(np.where(mask_one==1)[1]))//2 + np.min(np.where(mask_one==1)[1])
            
            cell_mask = cv2.putText(cell_mask, text, (text_y, text_x), font, font_scale, cell_color, thickness)
        except ZeroDivisionError:
            pass
            
    cell_mask = cv2.addWeighted(np.array(cell_mask, dtype=np.uint8), 1, image, 1, 0)
    return properties, cell_num + start_idx, cell_mask

def main():
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Process microscopy images and save results')
        parser.add_argument('-i', '--input', type=str, help='Path to the directory containing PNG images')
        parser.add_argument('-o', '--output', type=str, help='Path to the directory containing segmentation masks and cell properties')
        parser.add_argument('--crop_size', type=int, default=10000, help='Size value for cropping image')
        parser.add_argument('--model', type=str, default='tiny', choices=['tiny', 'small', 'base', 'large'],
                          help='Model size to use')
        args = parser.parse_args()
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config = MODEL_CONFIGS[args.model]
        sam2 = build_sam2(model_config['config'], model_config['checkpoint'], 
                         device=device, apply_postprocessing=False)
        mask_generator = SAM2AutomaticMaskGenerator(sam2)
        
        # Process files
        for file in os.listdir(args.input):
            if '.png' in file:
                fn = os.path.join(args.input, file)
                cell_properties = {}
                sta_time = time.time()
                
                try:
                    image = cv2.imread(fn)
                except:
                    image = imread(fn)
                
                cell_properties_patch, sta_idx, cell_mask_batch = patch_property(
                    image, 0, mask_generator)
                cell_properties.update(cell_properties_patch)
                
                cv2.imwrite(os.path.join(args.output, file.strip('.png') + '_cell_mask.png'),
                           cell_mask_batch)
                
                cell_properties = pd.DataFrame.from_dict(cell_properties, orient='index')
                cell_properties.to_csv(os.path.join(args.output, file.replace('.png', '.csv')))
                
                end_time = time.time()
                print(f'File {fn} completed, time used {(end_time - sta_time) / 60:.2f} min')
    else:
        # Launch GUI
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())

if __name__ == '__main__':
    import sys
    main() 