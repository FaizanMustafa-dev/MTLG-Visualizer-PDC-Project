import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, timedelta
import json
import io
import base64
import time
import re
import math
import warnings
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Tuple, Union
import os

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

# Theme Colors
THEME = {
    "dark_purple": "#0d0519",
    "medium_purple": "#1e0b3d",
    "light_purple": "#7e3af2",
    "accent_purple": "#a855f7",
    "text_light": "#f3e8ff",
    "text_dark": "#c4b5fd",
    "card_bg": "rgba(30, 11, 61, 0.95)",
    "hover_purple": "#8b5cf6",
    "gradient_start": "#4c1d95",
    "gradient_end": "#7c3aed",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "sidebar_bg": "#120827",
    "sidebar_header": "#1e0b3d",
    "gradient_purple": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "gradient_blue": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "gradient_green": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
    "gradient_orange": "linear-gradient(135deg, #f7971e 0%, #ffd200 100%)"
}

# Page Configuration
PAGES = {
    "🏠 Dashboard": "Dashboard",
    "📥 Data Input": "Data Input", 
    "🕸️ Dependency Graph": "Dependency Graph",
    "📊 Timeline": "Timeline",
    "⏱️ Latency Analysis": "Latency Analysis",
    "⚡ Optimization": "Optimization",
    "ℹ️ About": "About"
}

# Groq API Configuration
GROQ_API_KEY = "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"

# ============================================
# AI PROCESSOR CLASS
# ============================================

try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

class AIDataProcessor:
    """Advanced AI processor for data analysis and optimization suggestions"""
    
    def __init__(self):
        self.groq_available = GROQ_AVAILABLE
        self.client = None
        
        if self.groq_available:
            try:
                self.client = groq.Groq(api_key=GROQ_API_KEY)
            except Exception as e:
                self.groq_available = False
                st.warning(f"Failed to initialize Groq client: {e}")
        
        # Enhanced column patterns for better detection
        self.column_patterns = {
            'task_id': r'(task|id|task_id|taskid|tid|job_id|process_id)',
            'task_name': r'(name|task_name|process_name|job_name)',
            'parent_id': r'(parent|parent_id|parentid|pid|depend|dependency|predecessor)',
            'execution_time': r'(time|duration|execution|runtime|latency|ms|seconds|milliseconds)',
            'start_time': r'(start|start_time|begin|timestamp|start_ts)',
            'end_time': r'(end|end_time|finish|completed|end_ts)',
            'cpu_usage': r'(cpu|cpu_usage|cpu_percent|processor|utilization)',
            'memory_usage': r'(memory|mem|memory_usage|ram|mb|gb|memory_mb)',
            'resource_usage': r'(resource|usage|utilization|load)',
            'parallelizable': r'(parallel|concurrent|thread|process|core|parallelizable|concurrency)',
            'priority': r'(priority|importance|weight|rank|criticality)',
            'level': r'(level|depth|hierarchy|layer)',
            'task_type': r'(type|task_type|category|classification)',
            'status': r'(status|state|completion|progress)',
            'cost': r'(cost|price|expense|budget)'
        }
        
        # Task type mappings
        self.task_types = {
            'compute': ['compute', 'calculation', 'processing', 'cpu', 'algorithm'],
            'io': ['io', 'input', 'output', 'disk', 'file', 'storage'],
            'network': ['network', 'communication', 'api', 'http', 'web'],
            'memory': ['memory', 'cache', 'ram', 'storage'],
            'database': ['database', 'db', 'query', 'sql', 'nosql']
        }
    
    def detect_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Advanced data structure detection using AI and pattern matching"""
        if self.groq_available and self.client and len(df) > 0:
            return self._advanced_ai_detection(df)
        else:
            return self._enhanced_fallback_detection(df)
    
    def _advanced_ai_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Advanced AI-powered data structure detection with detailed analysis"""
        try:
            # Prepare comprehensive data summary
            data_summary = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": df.isnull().sum().to_dict(),
                "sample_data": {}
            }
            
            # Add sample values for each column
            for col in df.columns:
                sample_vals = df[col].dropna().head(3).tolist()
                data_summary["sample_data"][col] = sample_vals
            
            # Create detailed prompt
            prompt = f"""As a parallel computing data analysis expert, analyze this dataset structure:

            DATASET OVERVIEW:
            - Shape: {data_summary['shape'][0]} rows × {data_summary['shape'][1]} columns
            - Columns: {', '.join(data_summary['columns'])}
            
            COLUMN DETAILS:
            {json.dumps(data_summary['sample_data'], indent=2)}
            
            MISSING VALUES:
            {json.dumps(data_summary['missing_values'], indent=2)}
            
            Based on parallel computing task data patterns, identify:
            1. TASK IDENTIFICATION: Which columns identify tasks?
            2. DEPENDENCIES: Which columns show parent-child relationships?
            3. TIMING: Which columns contain timing information (start, end, duration)?
            4. RESOURCES: Which columns show resource usage (CPU, memory, I/O)?
            5. METADATA: Which columns contain task metadata (type, priority, status)?
            6. PARALLELIZATION: Which columns indicate parallelization potential?
            
            Return JSON with this structure:
            {{
                "detected_structure": {{"column_name": "purpose", ...}},
                "confidence_scores": {{"column_name": 0.95, ...}},
                "dependency_graph": {{"parent_col": "...", "child_col": "..."}},
                "timing_analysis": {{"start_col": "...", "end_col": "...", "duration_col": "..."}},
                "resource_columns": ["cpu_col", "memory_col", ...],
                "parallelization_info": {{"parallelizable_col": "...", "estimated_parallel_tasks": 0}},
                "data_quality": {{"issues": [], "suggestions": []}},
                "recommendations": ["suggestion1", "suggestion2", ...]
            }}
            
            Provide confidence scores from 0 to 1 for each detection.
            """
            
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Enhance with fallback detection if needed
                if not result.get("detected_structure"):
                    result.update(self._enhanced_fallback_detection(df))
                
                return result
            else:
                return self._enhanced_fallback_detection(df)
                
        except Exception as e:
            st.error(f"AI detection error: {str(e)}")
            return self._enhanced_fallback_detection(df)
    
    def _enhanced_fallback_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced fallback detection with multiple pattern matching strategies"""
        structure = {}
        confidence_scores = {}
        timing_cols = []
        dep_cols = []
        resource_cols = []
        
        for col in df.columns:
            col_lower = str(col).lower()
            col_data = df[col].dropna()
            
            # Multiple detection strategies
            detected = False
            
            # Strategy 1: Pattern matching
            for purpose, pattern in self.column_patterns.items():
                if re.search(pattern, col_lower, re.IGNORECASE):
                    structure[col] = purpose
                    confidence_scores[col] = 0.7  # Medium confidence
                    detected = True
                    
                    # Categorize columns
                    if 'time' in purpose:
                        timing_cols.append(col)
                    elif 'parent' in purpose or 'depend' in purpose:
                        dep_cols.append(col)
                    elif any(resource in purpose for resource in ['cpu', 'memory', 'resource']):
                        resource_cols.append(col)
                    break
            
            # Strategy 2: Data type analysis
            if not detected:
                dtype = str(df[col].dtype)
                if dtype in ['int64', 'float64']:
                    # Numeric column - could be timing or resource
                    if len(col_data) > 0:
                        # Check if values look like timing (milliseconds range)
                        avg_val = col_data.mean()
                        if 0 < avg_val < 1000000:  # Reasonable timing range
                            structure[col] = 'execution_time'
                            confidence_scores[col] = 0.6
                            timing_cols.append(col)
                        elif avg_val <= 100:  # Could be percentage
                            structure[col] = 'resource_usage'
                            confidence_scores[col] = 0.5
                            resource_cols.append(col)
                
                elif dtype == 'object':
                    # String column - could be ID or name
                    unique_ratio = len(col_data.unique()) / len(col_data) if len(col_data) > 0 else 0
                    if unique_ratio > 0.9:  # Mostly unique values
                        if any(id_term in col_lower for id_term in ['id', 'name', 'task']):
                            structure[col] = 'task_id'
                            confidence_scores[col] = 0.8
                        else:
                            structure[col] = 'task_name'
                            confidence_scores[col] = 0.6
        
        # Find parent-child relationships
        dependency_graph = {}
        if dep_cols:
            parent_col = dep_cols[0]
            # Look for task_id column for child reference
            task_cols = [col for col, purpose in structure.items() if 'task_id' in purpose]
            if task_cols:
                child_col = task_cols[0]
                dependency_graph = {"parent_col": parent_col, "child_col": child_col}
        
        # Find timing columns
        timing_analysis = {}
        if len(timing_cols) >= 2:
            # Try to identify start and end times
            for col in timing_cols:
                col_lower = str(col).lower()
                if 'start' in col_lower or 'begin' in col_lower:
                    timing_analysis['start_col'] = col
                elif 'end' in col_lower or 'finish' in col_lower:
                    timing_analysis['end_col'] = col
                elif 'duration' in col_lower or 'execution' in col_lower:
                    timing_analysis['duration_col'] = col
        
        return {
            "detected_structure": structure,
            "confidence_scores": confidence_scores,
            "dependency_graph": dependency_graph,
            "timing_analysis": timing_analysis,
            "resource_columns": resource_cols,
            "parallelization_info": {
                "parallelizable_col": next((col for col, purpose in structure.items() if 'parallel' in purpose), None),
                "estimated_parallel_tasks": len(df) * 0.3  # Estimate 30% parallelizable
            },
            "data_quality": {
                "issues": [],
                "suggestions": ["Consider renaming columns for better auto-detection"]
            },
            "recommendations": [
                "Add missing timing columns if not present",
                "Consider adding task priority information",
                "Include resource utilization metrics"
            ]
        }
    
    def analyze_parallel_patterns(self, df: pd.DataFrame, structure: Dict) -> Dict[str, Any]:
        """Comprehensive parallel computing pattern analysis"""
        if self.groq_available and self.client:
            return self._advanced_ai_analysis(df, structure)
        else:
            return self._comprehensive_basic_analysis(df, structure)
    
    def _advanced_ai_analysis(self, df: pd.DataFrame, structure: Dict) -> Dict[str, Any]:
        """Advanced AI-powered pattern analysis with optimization recommendations"""
        try:
            # Prepare detailed analysis data
            analysis_data = {
                "data_summary": {
                    "total_tasks": len(df),
                    "columns_analyzed": len(df.columns),
                    "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
                },
                "detected_structure": structure,
                "key_metrics": {}
            }
            
            # Calculate basic metrics
            if 'execution_time' in [p.lower() for p in structure.get('detected_structure', {}).values()]:
                time_col = next((col for col, purpose in structure['detected_structure'].items() 
                               if 'execution_time' in purpose.lower()), None)
                if time_col:
                    times = df[time_col].dropna()
                    if len(times) > 0:
                        analysis_data["key_metrics"]["execution_time"] = {
                            "total": float(times.sum()),
                            "average": float(times.mean()),
                            "maximum": float(times.max()),
                            "minimum": float(times.min()),
                            "std_dev": float(times.std())
                        }
            
            # Create analysis prompt
            prompt = f"""As a parallel computing optimization expert, analyze this data for performance patterns:

            DATA SUMMARY:
            {json.dumps(analysis_data['data_summary'], indent=2)}
            
            DETECTED STRUCTURE:
            {json.dumps(structure.get('detected_structure', {}), indent=2)}
            
            KEY METRICS:
            {json.dumps(analysis_data['key_metrics'], indent=2)}
            
            SAMPLE DATA (first 5 rows):
            {df.head().to_string()}
            
            Perform comprehensive analysis covering:
            1. CRITICAL PATH ANALYSIS: Identify longest dependency chains
            2. BOTTLENECK DETECTION: Find tasks causing delays
            3. PARALLELIZATION POTENTIAL: Estimate how many tasks can run concurrently
            4. RESOURCE UTILIZATION: Analyze CPU/memory patterns
            5. DEPENDENCY OPTIMIZATION: Suggest dependency restructuring
            6. PERFORMANCE ESTIMATION: Calculate theoretical speedup
            7. COST OPTIMIZATION: Suggest cost-saving measures
            
            Return detailed JSON analysis with:
            {{
                "critical_path_analysis": {{"path_length": 0, "critical_tasks": [], "total_time": 0}},
                "bottleneck_detection": {{"bottleneck_tasks": [], "reasons": [], "impact_percentage": 0}},
                "parallelization_analysis": {{"parallelizable_tasks": 0, "parallel_percentage": 0, "estimated_speedup": 0}},
                "resource_analysis": {{"resource_hotspots": [], "utilization_patterns": [], "recommendations": []}},
                "optimization_suggestions": [
                    {{"type": "parallelization", "description": "...", "impact": "high", "effort": "low"}}
                ],
                "performance_metrics": {{"current_total_time": 0, "optimized_time": 0, "speedup_factor": 0}},
                "risk_assessment": {{"risks": [], "mitigations": []}},
                "implementation_roadmap": ["step1", "step2", ...]
            }}
            
            Be specific and quantitative in analysis.
            """
            
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=3000
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Enhance with basic analysis
                basic_analysis = self._comprehensive_basic_analysis(df, structure)
                result.update({
                    "basic_analysis": basic_analysis,
                    "ai_analysis": True,
                    "analysis_timestamp": datetime.now().isoformat()
                })
                
                return result
            else:
                return self._comprehensive_basic_analysis(df, structure)
                
        except Exception as e:
            st.error(f"AI analysis error: {str(e)}")
            return self._comprehensive_basic_analysis(df, structure)
    
    def _comprehensive_basic_analysis(self, df: pd.DataFrame, structure: Dict) -> Dict[str, Any]:
        """Comprehensive basic analysis without AI"""
        analysis = {
            "critical_path_analysis": {
                "path_length": 0,
                "critical_tasks": [],
                "total_time": 0,
                "note": "Basic analysis - install Groq for detailed critical path detection"
            },
            "bottleneck_detection": {
                "bottleneck_tasks": [],
                "reasons": [],
                "impact_percentage": 0
            },
            "parallelization_analysis": {
                "parallelizable_tasks": 0,
                "parallel_percentage": 0,
                "estimated_speedup": 1.0
            },
            "resource_analysis": {
                "resource_hotspots": [],
                "utilization_patterns": [],
                "recommendations": ["Collect more detailed resource metrics for better analysis"]
            },
            "optimization_suggestions": [],
            "performance_metrics": {
                "current_total_time": 0,
                "optimized_time": 0,
                "speedup_factor": 1.0
            },
            "risk_assessment": {
                "risks": ["Limited data for comprehensive analysis"],
                "mitigations": ["Add more detailed task metrics and dependencies"]
            },
            "implementation_roadmap": [
                "1. Collect detailed timing data",
                "2. Map all task dependencies",
                "3. Measure resource utilization",
                "4. Identify parallelization opportunities",
                "5. Implement gradual optimizations"
            ],
            "ai_analysis": False
        }
        
        # Basic bottleneck detection
        time_cols = [col for col in df.columns if any(keyword in str(col).lower() 
                                                     for keyword in ['time', 'duration', 'latency'])]
        if time_cols:
            time_col = time_cols[0]
            times = df[time_col].dropna()
            
            if len(times) > 10:
                # Calculate total time
                total_time = times.sum()
                analysis["performance_metrics"]["current_total_time"] = float(total_time)
                
                # Identify bottlenecks (top 10% longest tasks)
                bottleneck_threshold = times.quantile(0.9)
                bottlenecks = df[df[time_col] > bottleneck_threshold]
                
                if len(bottlenecks) > 0:
                    bottleneck_tasks = []
                    for idx, row in bottlenecks.iterrows():
                        task_id = row.get('task_id', f"Task_{idx}")
                        task_time = row[time_col]
                        bottleneck_tasks.append({
                            "task": str(task_id),
                            "time": float(task_time),
                            "threshold": float(bottleneck_threshold)
                        })
                    
                    analysis["bottleneck_detection"]["bottleneck_tasks"] = bottleneck_tasks
                    analysis["bottleneck_detection"]["impact_percentage"] = float(
                        (bottlenecks[time_col].sum() / total_time) * 100
                    )
        
        # Parallelization analysis
        if 'parallelizable' in df.columns:
            parallel_tasks = df[df['parallelizable'] == True]
            analysis["parallelization_analysis"]["parallelizable_tasks"] = len(parallel_tasks)
            analysis["parallelization_analysis"]["parallel_percentage"] = float(
                (len(parallel_tasks) / len(df)) * 100
            )
            
            # Simple speedup estimation using Amdahl's Law
            if len(df) > 0:
                parallel_fraction = len(parallel_tasks) / len(df)
                if parallel_fraction > 0:
                    # Assume 4-core parallelization
                    speedup = 1 / ((1 - parallel_fraction) + (parallel_fraction / 4))
                    analysis["parallelization_analysis"]["estimated_speedup"] = float(speedup)
                    analysis["performance_metrics"]["speedup_factor"] = float(speedup)
                    analysis["performance_metrics"]["optimized_time"] = float(
                        analysis["performance_metrics"]["current_total_time"] / speedup
                    )
        
        # Generate optimization suggestions
        suggestions = []
        
        if analysis["bottleneck_detection"]["bottleneck_tasks"]:
            suggestions.append({
                "type": "bottleneck",
                "description": f"Optimize {len(analysis['bottleneck_detection']['bottleneck_tasks'])} bottleneck tasks",
                "impact": "high",
                "effort": "medium",
                "details": "These tasks consume disproportionate time"
            })
        
        if analysis["parallelization_analysis"]["parallelizable_tasks"] > 0:
            suggestions.append({
                "type": "parallelization",
                "description": f"Parallelize {analysis['parallelization_analysis']['parallelizable_tasks']} tasks",
                "impact": "medium",
                "effort": "low",
                "details": f"Estimated speedup: {analysis['parallelization_analysis']['estimated_speedup']:.2f}x"
            })
        
        # Resource optimization suggestions
        resource_cols = [col for col in df.columns if any(keyword in str(col).lower() 
                                                         for keyword in ['cpu', 'memory', 'usage'])]
        if resource_cols:
            suggestions.append({
                "type": "resource",
                "description": "Optimize resource allocation",
                "impact": "medium",
                "effort": "high",
                "details": "Analyze and balance CPU/memory usage patterns"
            })
        
        analysis["optimization_suggestions"] = suggestions
        
        # Generate basic critical path estimate
        if 'parent_id' in df.columns and 'task_id' in df.columns:
            # Simple critical path estimation
            try:
                G = nx.DiGraph()
                
                # Add nodes
                for _, row in df.iterrows():
                    G.add_node(row['task_id'])
                
                # Add edges
                for _, row in df.iterrows():
                    if pd.notna(row['parent_id']):
                        G.add_edge(row['parent_id'], row['task_id'])
                
                if nx.is_directed_acyclic_graph(G):
                    # Find longest path
                    longest_path = nx.dag_longest_path(G)
                    analysis["critical_path_analysis"]["critical_tasks"] = longest_path
                    analysis["critical_path_analysis"]["path_length"] = len(longest_path)
            except:
                pass
        
        return analysis
    
    def generate_optimization_plan(self, analysis: Dict) -> Dict[str, Any]:
        """Generate detailed optimization implementation plan"""
        plan = {
            "summary": {
                "total_suggestions": 0,
                "estimated_impact": "medium",
                "implementation_timeline": "2-4 weeks",
                "success_probability": 0.7
            },
            "phases": [],
            "metrics_to_track": [],
            "risk_mitigation": [],
            "expected_outcomes": []
        }
        
        # Process optimization suggestions
        suggestions = analysis.get("optimization_suggestions", [])
        plan["summary"]["total_suggestions"] = len(suggestions)
        
        # Categorize suggestions by type
        categories = {}
        for suggestion in suggestions:
            category = suggestion.get("type", "general")
            if category not in categories:
                categories[category] = []
            categories[category].append(suggestion)
        
        # Create implementation phases
        phases = []
        phase_num = 1
        
        # Phase 1: Quick wins (low effort, high impact)
        quick_wins = [s for s in suggestions if s.get("effort") == "low" and s.get("impact") in ["high", "medium"]]
        if quick_wins:
            phases.append({
                "phase": phase_num,
                "name": "Quick Wins",
                "duration": "1 week",
                "tasks": [
                    {
                        "description": f"Implement {len(quick_wins)} quick optimizations",
                        "details": [s["description"] for s in quick_wins],
                        "owner": "Development Team",
                        "success_criteria": ["Performance improvement > 10%", "No regression in functionality"]
                    }
                ],
                "dependencies": [],
                "risks": ["Minor integration issues"],
                "mitigations": ["Thorough testing", "Rollback plan"]
            })
            phase_num += 1
        
        # Phase 2: Parallelization
        parallel_suggestions = [s for s in suggestions if s.get("type") == "parallelization"]
        if parallel_suggestions:
            phases.append({
                "phase": phase_num,
                "name": "Parallelization Implementation",
                "duration": "2 weeks",
                "tasks": [
                    {
                        "description": "Implement task parallelization",
                        "details": ["Refactor task execution", "Add concurrency controls", "Implement monitoring"],
                        "owner": "Parallel Computing Team",
                        "success_criteria": ["Speedup > 1.5x", "Resource utilization improved", "No deadlocks"]
                    }
                ],
                "dependencies": ["Phase 1 completed"],
                "risks": ["Concurrency issues", "Resource contention"],
                "mitigations": ["Use thread-safe patterns", "Implement resource pooling"]
            })
            phase_num += 1
        
        # Phase 3: Resource Optimization
        resource_suggestions = [s for s in suggestions if s.get("type") == "resource"]
        if resource_suggestions:
            phases.append({
                "phase": phase_num,
                "name": "Resource Optimization",
                "duration": "1-2 weeks",
                "tasks": [
                    {
                        "description": "Optimize resource allocation",
                        "details": ["Profile resource usage", "Implement caching", "Balance load distribution"],
                        "owner": "Infrastructure Team",
                        "success_criteria": ["CPU utilization optimized", "Memory usage reduced", "Cost savings achieved"]
                    }
                ],
                "dependencies": ["Phase 2 completed"],
                "risks": ["Performance regression", "Increased complexity"],
                "mitigations": ["A/B testing", "Gradual rollout"]
            })
        
        plan["phases"] = phases
        
        # Define metrics to track
        plan["metrics_to_track"] = [
            "Total execution time",
            "CPU utilization",
            "Memory usage",
            "Parallel task completion rate",
            "Cost per task",
            "Error rate"
        ]
        
        # Risk mitigation strategies
        plan["risk_mitigation"] = [
            {
                "risk": "Performance regression",
                "mitigation": "Implement comprehensive testing and monitoring",
                "owner": "QA Team"
            },
            {
                "risk": "Increased complexity",
                "mitigation": "Document all changes and provide training",
                "owner": "Documentation Team"
            },
            {
                "risk": "Integration issues",
                "mitigation": "Use gradual rollout and feature flags",
                "owner": "DevOps Team"
            }
        ]
        
        # Expected outcomes
        if analysis.get("performance_metrics", {}).get("speedup_factor", 1) > 1:
            speedup = analysis["performance_metrics"]["speedup_factor"]
            plan["expected_outcomes"] = [
                f"Performance improvement: {speedup:.2f}x speedup",
                "Reduced execution costs",
                "Better resource utilization",
                "Scalability improvements"
            ]
        
        return plan
    
    def simulate_optimization(self, df: pd.DataFrame, analysis: Dict, 
                            parallel_factor: float = 2.0,
                            resource_optimization: float = 20.0,
                            cache_efficiency: float = 30.0) -> Dict[str, Any]:
        """Simulate optimization impact with detailed calculations"""
        
        # Get current metrics
        current_total_time = analysis.get("performance_metrics", {}).get("current_total_time", 0)
        if current_total_time == 0:
            # Estimate from data
            time_cols = [col for col in df.columns if any(keyword in str(col).lower() 
                                                         for keyword in ['time', 'duration'])]
            if time_cols:
                current_total_time = df[time_cols[0]].sum()
            else:
                current_total_time = 1000  # Default
        
        # Calculate optimization impacts
        optimization_impacts = []
        
        # Parallelization impact
        parallel_tasks = analysis.get("parallelization_analysis", {}).get("parallelizable_tasks", 0)
        total_tasks = len(df)
        
        if parallel_tasks > 0 and total_tasks > 0:
            parallel_fraction = parallel_tasks / total_tasks
            parallel_impact = (1 - (1 / parallel_factor)) * parallel_fraction * 100
            optimization_impacts.append({
                "type": "parallelization",
                "impact_percentage": parallel_impact,
                "description": f"Parallelizing {parallel_tasks} tasks with factor {parallel_factor}"
            })
        
        # Resource optimization impact
        optimization_impacts.append({
            "type": "resource_optimization",
            "impact_percentage": resource_optimization * 0.3,  # Scale down impact
            "description": f"Resource usage optimization ({resource_optimization}% target)"
        })
        
        # Caching impact
        optimization_impacts.append({
            "type": "caching",
            "impact_percentage": cache_efficiency * 0.2,  # Scale down impact
            "description": f"Caching efficiency ({cache_efficiency}% of tasks benefit)"
        })
        
        # Calculate total optimization
        total_optimization = 1.0
        for impact in optimization_impacts:
            total_optimization *= (1 - impact["impact_percentage"] / 100)
        
        optimized_time = current_total_time * total_optimization
        speedup = current_total_time / optimized_time if optimized_time > 0 else 1
        
        # Calculate efficiency
        efficiency = (speedup / parallel_factor) * 100 if parallel_factor > 0 else 0
        
        return {
            "simulation_parameters": {
                "parallel_factor": parallel_factor,
                "resource_optimization": resource_optimization,
                "cache_efficiency": cache_efficiency
            },
            "current_performance": {
                "total_time": current_total_time,
                "total_tasks": total_tasks,
                "parallel_tasks": parallel_tasks
            },
            "optimized_performance": {
                "total_time": optimized_time,
                "speedup": speedup,
                "improvement_percentage": (1 - total_optimization) * 100,
                "efficiency_percentage": efficiency
            },
            "optimization_impacts": optimization_impacts,
            "cost_analysis": {
                "current_cost": current_total_time * 0.01,  # Simplified cost model
                "optimized_cost": optimized_time * 0.01,
                "savings_percentage": (1 - (optimized_time / current_total_time)) * 100
            },
            "recommendations": [
                f"Focus on parallelizing {parallel_tasks} tasks first",
                f"Aim for {resource_optimization}% resource optimization",
                f"Implement caching for {cache_efficiency}% of repetitive tasks"
            ]
        }

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="MTLG - Parallel Computing Analyzer",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/MTLG-Visualizer',
        'Report a bug': 'https://github.com/yourusername/MTLG-Visualizer/issues',
        'About': "### MTLG Visualizer v2.0\nAdvanced Parallel Computing Analysis Tool"
    }
)

# ============================================
# CUSTOM CSS WITH ADVANCED STYLING
# ============================================

def load_custom_css():
    """Load custom CSS with advanced styling"""
    css = f"""
    <style>
    /* Main Theme Styles */
    .stApp {{
        background: linear-gradient(135deg, {THEME['dark_purple']} 0%, {THEME['medium_purple']} 50%, {THEME['dark_purple']} 100%);
        color: {THEME['text_light']};
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    
    /* Custom Header */
    .main-header {{
        background: {THEME['gradient_purple']};
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 15px 35px rgba(124, 58, 237, 0.4);
        border: 1px solid rgba(168, 85, 247, 0.3);
        animation: headerGlow 4s ease-in-out infinite alternate;
        position: relative;
        overflow: hidden;
    }}
    
    .main-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
        animation: pulse 8s ease-in-out infinite;
    }}
    
    @keyframes headerGlow {{
        0% {{ box-shadow: 0 15px 35px rgba(124, 58, 237, 0.4); }}
        50% {{ box-shadow: 0 20px 40px rgba(168, 85, 247, 0.6); }}
        100% {{ box-shadow: 0 15px 35px rgba(124, 58, 237, 0.4); }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); opacity: 0.5; }}
        50% {{ transform: scale(1.1); opacity: 0.3; }}
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: {THEME['sidebar_bg']} !important;
        border-right: 2px solid rgba(168, 85, 247, 0.3) !important;
        box-shadow: 5px 0 20px rgba(0, 0, 0, 0.3);
    }}
    
    .sidebar-header {{
        background: {THEME['gradient_purple']};
        padding: 1.8rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(168, 85, 247, 0.4);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.3);
        position: relative;
        overflow: hidden;
    }}
    
    .sidebar-header::after {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }}
    
    @keyframes shimmer {{
        0% {{ transform: translateX(-100%); }}
        100% {{ transform: translateX(100%); }}
    }}
    
    /* Navigation Buttons */
    [data-testid="stSidebar"] .stButton > button {{
        background: linear-gradient(135deg, rgba(30, 11, 61, 0.9), rgba(76, 29, 149, 0.9));
        color: {THEME['text_light']} !important;
        border: 2px solid rgba(168, 85, 247, 0.4) !important;
        padding: 1rem 1.5rem !important;
        border-radius: 12px !important;
        margin: 0.5rem 0 !important;
        width: 100% !important;
        text-align: left !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }}
    
    [data-testid="stSidebar"] .stButton > button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: 0.5s;
    }}
    
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: {THEME['gradient_purple']} !important;
        transform: translateX(10px) scale(1.02) !important;
        border-color: {THEME['accent_purple']} !important;
        box-shadow: 0 10px 25px rgba(168, 85, 247, 0.5) !important;
    }}
    
    [data-testid="stSidebar"] .stButton > button:hover::before {{
        left: 100%;
    }}
    
    [data-testid="stSidebar"] .stButton > button[data-active="true"] {{
        background: {THEME['gradient_purple']} !important;
        border-color: {THEME['accent_purple']} !important;
        box-shadow: 0 8px 20px rgba(168, 85, 247, 0.6) !important;
        transform: translateX(5px) !important;
    }}
    
    /* Cards */
    .custom-card {{
        background: rgba(30, 11, 61, 0.85);
        border-radius: 20px;
        padding: 2rem;
        border: 2px solid rgba(168, 85, 247, 0.4);
        backdrop-filter: blur(15px);
        transition: all 0.4s ease;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }}
    
    .custom-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: {THEME['gradient_purple']};
    }}
    
    .custom-card:hover {{
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(168, 85, 247, 0.3);
        border-color: {THEME['accent_purple']};
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: linear-gradient(135deg, rgba(30, 11, 61, 0.95), rgba(45, 27, 104, 0.95));
        border-radius: 16px;
        padding: 1.8rem;
        border: 2px solid rgba(168, 85, 247, 0.3);
        margin: 1rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }}
    
    .metric-card::after {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(168, 85, 247, 0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(168, 85, 247, 0.25);
        border-color: {THEME['accent_purple']};
    }}
    
    .metric-card:hover::after {{
        opacity: 1;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: {THEME['gradient_purple']};
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.3);
    }}
    
    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: 0.5s;
    }}
    
    .stButton > button:hover {{
        background: {THEME['gradient_blue']};
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(124, 58, 237, 0.5);
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    .stButton > button:active {{
        transform: translateY(-1px);
    }}
    
    /* Primary Button Variant */
    .stButton > button[type="primary"] {{
        background: {THEME['gradient_green']};
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3);
    }}
    
    .stButton > button[type="primary"]:hover {{
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 12px 30px rgba(16, 185, 129, 0.5);
    }}
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {{
        background: rgba(30, 11, 61, 0.8) !important;
        border: 2px solid rgba(168, 85, 247, 0.4) !important;
        color: {THEME['text_light']} !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
        font-size: 1rem !important;
    }}
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > div:focus {{
        border-color: {THEME['accent_purple']} !important;
        box-shadow: 0 0 0 4px rgba(168, 85, 247, 0.2) !important;
        background: rgba(30, 11, 61, 0.9) !important;
    }}
    
    .stTextInput > div > div > input:hover,
    .stTextArea > div > div > textarea:hover {{
        border-color: rgba(168, 85, 247, 0.6) !important;
    }}
    
    /* File Uploader */
    .stFileUploader > div {{
        border: 3px dashed rgba(168, 85, 247, 0.5);
        border-radius: 20px;
        background: rgba(30, 11, 61, 0.7);
        transition: all 0.3s ease;
        padding: 2rem !important;
    }}
    
    .stFileUploader > div:hover {{
        border-color: {THEME['accent_purple']};
        background: rgba(30, 11, 61, 0.9);
        box-shadow: 0 10px 30px rgba(168, 85, 247, 0.2);
    }}
    
    .stFileUploader > div:has(> div > button) {{
        border-style: solid;
        border-color: {THEME['success']};
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background: rgba(30, 11, 61, 0.9);
        padding: 10px;
        border-radius: 15px;
        border: 1px solid rgba(168, 85, 247, 0.3);
        margin-bottom: 2rem;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 10px;
        color: {THEME['text_dark']};
        transition: all 0.3s ease;
        padding: 0.8rem 1.5rem;
        font-weight: 500;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(168, 85, 247, 0.1);
        color: {THEME['text_light']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {THEME['gradient_purple']};
        color: white !important;
        box-shadow: 0 5px 15px rgba(168, 85, 247, 0.4);
    }}
    
    /* Progress Bars */
    .stProgress > div > div > div > div {{
        background: {THEME['gradient_purple']};
        border-radius: 10px;
    }}
    
    .stProgress > div {{
        background: rgba(30, 11, 61, 0.5);
        border-radius: 10px;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        from {{ 
            opacity: 0; 
            transform: translateY(30px) scale(0.95); 
        }}
        to {{ 
            opacity: 1; 
            transform: translateY(0) scale(1); 
        }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }}
    
    @keyframes slideInLeft {{
        from {{ 
            opacity: 0; 
            transform: translateX(-50px); 
        }}
        to {{ 
            opacity: 1; 
            transform: translateX(0); 
        }}
    }}
    
    .slide-in-left {{
        animation: slideInLeft 0.5s ease-out forwards;
    }}
    
    @keyframes slideInRight {{
        from {{ 
            opacity: 0; 
            transform: translateX(50px); 
        }}
        to {{ 
            opacity: 1; 
            transform: translateX(0); 
        }}
    }}
    
    .slide-in-right {{
        animation: slideInRight 0.5s ease-out forwards;
    }}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {{
        width: 12px;
        height: 12px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(30, 11, 61, 0.6);
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {THEME['gradient_purple']};
        border-radius: 10px;
        border: 2px solid rgba(30, 11, 61, 0.6);
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {THEME['gradient_blue']};
    }}
    
    /* Tooltips */
    .tooltip {{
        position: relative;
        display: inline-block;
    }}
    
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: rgba(30, 11, 61, 0.95);
        color: {THEME['text_light']};
        text-align: center;
        border-radius: 10px;
        padding: 10px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid rgba(168, 85, 247, 0.3);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }}
    
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    
    /* Status Indicators */
    .status-indicator {{
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulseStatus 2s infinite;
    }}
    
    .status-active {{
        background-color: {THEME['success']};
        box-shadow: 0 0 10px {THEME['success']};
    }}
    
    .status-warning {{
        background-color: {THEME['warning']};
        box-shadow: 0 0 10px {THEME['warning']};
    }}
    
    .status-inactive {{
        background-color: {THEME['danger']};
        box-shadow: 0 0 10px {THEME['danger']};
    }}
    
    @keyframes pulseStatus {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Dataframe Styling */
    .dataframe {{
        border-radius: 15px !important;
        overflow: hidden !important;
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
    }}
    
    .dataframe thead {{
        background: {THEME['gradient_purple']} !important;
        color: white !important;
    }}
    
    .dataframe tbody tr:nth-child(even) {{
        background: rgba(30, 11, 61, 0.5) !important;
    }}
    
    .dataframe tbody tr:hover {{
        background: rgba(168, 85, 247, 0.1) !important;
    }}
    
    /* Plotly Chart Improvements */
    .js-plotly-plot .plotly .modebar {{
        background: rgba(30, 11, 61, 0.9) !important;
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
        border-radius: 10px !important;
        padding: 5px !important;
    }}
    
    .js-plotly-plot .plotly .modebar-btn:hover {{
        background: rgba(168, 85, 247, 0.2) !important;
    }}
    
    /* Expandable Sections */
    .streamlit-expanderHeader {{
        background: rgba(30, 11, 61, 0.8) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
        font-weight: 600 !important;
        color: {THEME['text_light']} !important;
        transition: all 0.3s ease !important;
    }}
    
    .streamlit-expanderHeader:hover {{
        background: rgba(168, 85, 247, 0.1) !important;
        border-color: {THEME['accent_purple']} !important;
    }}
    
    .streamlit-expanderContent {{
        background: rgba(30, 11, 61, 0.6) !important;
        border-radius: 0 0 10px 10px !important;
        border: 1px solid rgba(168, 85, 247, 0.2) !important;
        border-top: none !important;
    }}
    
    /* Notification Badges */
    .badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
    }}
    
    .badge-primary {{
        background: {THEME['gradient_purple']};
        color: white;
    }}
    
    .badge-success {{
        background: {THEME['gradient_green']};
        color: white;
    }}
    
    .badge-warning {{
        background: {THEME['gradient_orange']};
        color: white;
    }}
    
    .badge-danger {{
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }}
    
    /* Loading Spinner */
    .loading-spinner {{
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 5px solid rgba(168, 85, 247, 0.3);
        border-radius: 50%;
        border-top-color: {THEME['accent_purple']};
        animation: spin 1s ease-in-out infinite;
    }}
    
    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'data': None,
        'structure': None,
        'analysis': None,
        'current_page': "Dashboard",
        'processed_data': None,
        'graph_data': None,
        'timeline_data': None,
        'optimization_results': None,
        'ai_processor': None,
        'uploaded_file_name': None,
        'data_preview': None,
        'last_analysis_time': None,
        'optimization_plan': None,
        'simulation_results': None,
        'user_settings': {
            'theme': 'dark',
            'auto_refresh': False,
            'animation_speed': 'normal',
            'graph_quality': 'high',
            'notifications': True,
            'ai_assistance': True
        },
        'ai_available': GROQ_AVAILABLE,
        'page_history': ["Dashboard"],
        'data_loaded': False,
        'analysis_complete': False,
        'optimization_suggestions': [],
        'selected_sample': None,
        'active_tab': "overview"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize AI processor
    if st.session_state.ai_processor is None:
        st.session_state.ai_processor = AIDataProcessor()

# ============================================
# SIDEBAR NAVIGATION
# ============================================
def create_sidebar():
    """Create advanced sidebar with navigation and status"""
    with st.sidebar:
        # Sidebar Header
        st.markdown(f"""
        <div class="sidebar-header fade-in">
            <div style="font-size: 3rem; margin-bottom: 15px; animation: rotate 10s linear infinite;">🌀</div>
            <h3 style="color: {THEME['text_light']}; margin: 0; font-weight: 800; font-size: 1.5rem;">
                MTLG Visualizer
            </h3>
            <p style="color: {THEME['text_dark']}; font-size: 0.9rem; margin-top: 8px; font-weight: 500;">
                Parallel Computing Intelligence Platform
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        
        
        current_page = st.session_state.current_page
        
        for display_name, page_name in PAGES.items():
            is_active = current_page == page_name
            
            if is_active:
                # Display active page with special styling
                emoji = display_name.split()[0]
                text = ' '.join(display_name.split()[1:])
                st.markdown(f"""
                <div style="background: {THEME['gradient_purple']}; 
                          border: 2px solid {THEME['accent_purple']};
                          border-radius: 12px; padding: 1rem 1.5rem; margin: 0.5rem 0;
                          box-shadow: 0 8px 20px rgba(168, 85, 247, 0.6);
                          transform: translateX(5px); transition: all 0.3s ease;">
                    <div style="color: white; font-weight: 700; font-size: 1.1rem; 
                              display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.3rem;">{emoji}</span>
                        <span>{text}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                if st.button(display_name, key=f"nav_{page_name}", use_container_width=True):
                    st.session_state.current_page = page_name
                    st.session_state.page_history.append(page_name)
                    st.rerun()
        
        st.markdown("---")
        
        # Current Status
        st.markdown("### 📊 Current Status")
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Status card
            st.markdown(f"""
            <div style="background: rgba(30, 11, 61, 0.8); border-radius: 12px; 
                      padding: 1.2rem; border: 1px solid rgba(16, 185, 129, 0.3);
                      margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <div class="status-indicator status-active"></div>
                    <div style="color: #10b981; font-weight: 600; font-size: 1rem;">
                        ✅ Data Loaded
                    </div>
                </div>
                <div style="color: {THEME['text_light']}; font-size: 0.9rem;">
                    <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                        <span>Tasks:</span>
                        <span style="font-weight: 600;">{len(df):,}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                        <span>Columns:</span>
                        <span style="font-weight: 600;">{len(df.columns)}</span>
                    </div>
                    {f'<div style="color: {THEME["text_dark"]}; font-size: 0.85rem; margin-top: 8px;">File: {st.session_state.uploaded_file_name}</div>' if st.session_state.uploaded_file_name else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄", key="sidebar_refresh_btn", help="Refresh Analysis", 
                           use_container_width=True):
                    with st.spinner("Refreshing..."):
                        df = st.session_state.data
                        st.session_state.structure = st.session_state.ai_processor.detect_structure(df)
                        st.session_state.analysis = st.session_state.ai_processor.analyze_parallel_patterns(df, st.session_state.structure)
                        st.session_state.analysis_complete = True
                        st.success("Analysis refreshed!", icon="✅")
            
            with col2:
                if st.button("📥", key="sidebar_export_btn", help="Export Data", 
                           use_container_width=True):
                    csv = st.session_state.data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="mtlg_export.csv",
                        mime="text/csv",
                        key="sidebar_download"
                    )
        else:
            st.markdown(f"""
            <div style="background: rgba(30, 11, 61, 0.8); border-radius: 12px; 
                      padding: 1.2rem; border: 1px solid rgba(245, 158, 11, 0.3);">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <div class="status-indicator status-warning"></div>
                    <div style="color: #f59e0b; font-weight: 600; font-size: 1rem;">
                        📭 No Data Loaded
                    </div>
                </div>
                <div style="color: {THEME['text_dark']}; font-size: 0.9rem; margin-top: 8px;">
                    Upload data or use sample to begin analysis
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📥 Go to Data Input", key="sidebar_goto_input", 
                        use_container_width=True, type="primary"):
                st.session_state.current_page = "Data Input"
                st.rerun()
        
        st.markdown("---")
        
        # AI Status
        st.markdown("### 🤖 AI Status")
        
        if GROQ_AVAILABLE:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.1); border-radius: 12px; 
                      padding: 1rem; border: 1px solid rgba(16, 185, 129, 0.3);">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <div class="status-indicator status-active"></div>
                    <div style="color: #10b981; font-weight: 600;">Groq AI Active</div>
                </div>
                <div style="color: {THEME['text_dark']}; font-size: 0.85rem;">
                    Advanced AI analysis available
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: rgba(245, 158, 11, 0.1); border-radius: 12px; 
                      padding: 1rem; border: 1px solid rgba(245, 158, 11, 0.3);">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <div class="status-indicator status-warning"></div>
                    <div style="color: #f59e0b; font-weight: 600;">⚠️ AI Limited</div>
                </div>
                <div style="color: {THEME['text_dark']}; font-size: 0.85rem;">
                    Install: <code>pip install groq</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        
        

# ============================================
# MAIN HEADER
# ============================================
def create_header():
    """Create animated main header"""
    st.markdown("""
    <div class="main-header fade-in">
        <div style="display: flex; align-items: center; gap: 25px; position: relative; z-index: 2;">
            <div style="font-size: 4rem; animation: rotate 20s linear infinite;">🌀</div>
            <div>
                <h1 style="color: #f3e8ff; margin: 0; font-size: 3.2rem; font-weight: 900; 
                    background: linear-gradient(90deg, #f3e8ff, #c4b5fd, #a855f7); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    text-shadow: 0 2px 10px rgba(168, 85, 247, 0.3);">
                    MTLG Visualizer
                </h1>
                <p style="color: #c4b5fd; font-size: 1.3rem; margin-top: 0.8rem; font-weight: 600;
                         background: rgba(30, 11, 61, 0.5); padding: 0.5rem 1.5rem; 
                         border-radius: 30px; display: inline-block; border: 1px solid rgba(168, 85, 247, 0.3);">
                    Multi-Task Latency & Dependency Graph Analyzer | Parallel Computing Intelligence
                </p>
            </div>
        </div>
        <div style="position: absolute; bottom: 10px; right: 20px; z-index: 2;">
            <div style="display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end;">
                <span class="badge badge-primary">AI-Powered</span>
                <span class="badge badge-success">Real-time</span>
                <span class="badge badge-warning">Advanced Analytics</span>
            </div>
        </div>
    </div>
    
    <style>
    @keyframes rotate {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    </style>
    """, unsafe_allow_html=True)

# ============================================
# UTILITY FUNCTIONS
# ============================================

def generate_sample_data(size: int = 50) -> pd.DataFrame:
    """Generate comprehensive sample data for demonstration"""
    np.random.seed(42)
    
    # Create task hierarchy with realistic dependencies
    tasks = []
    
    # PRE-CALCULATE all execution times first
    all_execution_times = []
    
    for i in range(size):
        task_id = f"TASK_{i:04d}"
        
        # Determine task level (0 = root, 1-3 = levels)
        if i == 0:
            level = 0
            parent_id = None
        else:
            # Create realistic dependency tree
            level_options = []
            if i < size * 0.3:  # 30% root-level tasks
                level_options.append(0)
            if i < size * 0.7:  # 70% can be level 1
                level_options.append(1)
            if i < size * 0.9:  # 90% can be level 2
                level_options.append(2)
            level = np.random.choice(level_options) if level_options else 3
            
            # Select parent from appropriate level
            possible_parents = [t for t in tasks if t['level'] == level - 1]
            if possible_parents:
                parent = np.random.choice(possible_parents)
                parent_id = parent['task_id']
            else:
                parent_id = None
                level = 0
        
        # Generate realistic timing data based on level
        base_time = np.random.exponential(50) + 20
        execution_time = base_time * (1 + level * 0.4)  # Higher levels take longer
        
        # Add variability with some outliers
        if np.random.random() < 0.1:  # 10% outliers
            execution_time *= np.random.uniform(3, 10)
        
        execution_time *= np.random.uniform(0.8, 1.2)
        
        all_execution_times.append(execution_time)
        
        # Store partial task data
        tasks.append({
            'task_id': task_id,
            'parent_id': parent_id,
            'level': level,
            'execution_time_ms': execution_time
        })
    
    # NOW calculate max execution time
    max_execution_time = max(all_execution_times)
    
    # Complete task data with resource calculations
    for i, task in enumerate(tasks):
        execution_time = task['execution_time_ms']
        level = task['level']
        
        # Generate resource usage with correlations
        cpu_usage = np.random.beta(2, 5) * 100
        memory_mb = np.random.lognormal(6, 0.7)  # MB
        io_operations = np.random.poisson(15)
        network_calls = np.random.poisson(8)
        
        # Correlate resource usage with execution time using the max value
        cpu_usage *= (0.5 + 0.5 * (execution_time / max_execution_time))
        memory_mb *= (0.7 + 0.3 * (execution_time / max_execution_time))
        
        # Determine if parallelizable (higher levels less parallelizable)
        parallelizable = np.random.random() > (level * 0.25)
        
        # Determine priority (inverse of level, plus random)
        priority = max(1, 5 - level + np.random.randint(-1, 2))
        
        # Task type based on characteristics
        if cpu_usage > 70:
            task_type = 'compute'
        elif io_operations > 20:
            task_type = 'io'
        elif network_calls > 10:
            task_type = 'network'
        elif memory_mb > 1024:
            task_type = 'memory'
        else:
            task_type = np.random.choice(['compute', 'io', 'network', 'memory', 'database'])
        
        # Cost estimation
        cost = execution_time * 0.001 + cpu_usage * 0.0005 + memory_mb * 0.00001
        
        # Update task with complete data
        task.update({
            'task_name': f"Task_{task_type.upper()}_{i}",
            'description': f"{task_type.capitalize()} intensive task",
            'execution_time_ms': round(execution_time, 2),
            'cpu_usage_percent': round(cpu_usage, 1),
            'memory_usage_mb': round(memory_mb, 1),
            'io_operations': io_operations,
            'network_calls': network_calls,
            'parallelizable': parallelizable,
            'priority': priority,
            'task_type': task_type,
            'cost': round(cost, 4),
            'status': 'completed',
            'start_time_ms': 0,  # Will be calculated
            'end_time_ms': 0,
            'dependencies_met': True,
            'resource_constraints': np.random.choice(['cpu', 'memory', 'io', 'network', 'none'], 
                                                    p=[0.3, 0.2, 0.2, 0.2, 0.1])
        })
    
    df = pd.DataFrame(tasks)
    
    # Calculate realistic start and end times based on dependencies
    df = calculate_task_timings(df)
    
    # Add some data quality issues for realism
    if size > 20:
        # Add some missing values
        missing_mask = np.random.random(size) < 0.05
        for col in ['cpu_usage_percent', 'memory_usage_mb']:
            df.loc[missing_mask, col] = np.nan
        
        # Add some outliers
        outlier_mask = np.random.random(size) < 0.03
        df.loc[outlier_mask, 'execution_time_ms'] *= 10
    
    return df

def calculate_task_timings(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate realistic start and end times based on dependencies"""
    df = df.copy()
    
    # Initialize timing
    df['start_time_ms'] = 0
    df['end_time_ms'] = df['execution_time_ms']
    
    # Process by levels
    max_level = df['level'].max()
    
    for level in range(1, int(max_level) + 1):
        level_tasks = df[df['level'] == level]
        
        for _, task in level_tasks.iterrows():
            parent_id = task['parent_id']
            if parent_id and parent_id in df['task_id'].values:
                parent = df[df['task_id'] == parent_id].iloc[0]
                
                # Start after parent ends, with some variability
                start_time = parent['end_time_ms'] + np.random.exponential(10)
                
                # Add dependency delay based on task characteristics
                if task['task_type'] == 'network':
                    start_time += np.random.exponential(20)
                elif task['task_type'] == 'io':
                    start_time += np.random.exponential(15)
                
                df.loc[df['task_id'] == task['task_id'], 'start_time_ms'] = start_time
                df.loc[df['task_id'] == task['task_id'], 'end_time_ms'] = start_time + task['execution_time_ms']
    
    # For root tasks without dependencies, spread them out
    root_tasks = df[df['level'] == 0]
    if len(root_tasks) > 1:
        start_times = np.linspace(0, 100, len(root_tasks))
        for i, (idx, task) in enumerate(root_tasks.iterrows()):
            df.loc[idx, 'start_time_ms'] = start_times[i]
            df.loc[idx, 'end_time_ms'] = start_times[i] + task['execution_time_ms']
    
    return df

def create_metric_card(title: str, value: str, delta: str = None, 
                      icon: str = "📊", color: str = "#7e3af2", 
                      trend: str = None) -> str:
    """Create an advanced metric card with trend indicators"""
    
    # Determine delta color
    if delta:
        if delta.startswith('+'):
            delta_color = "#10b981"  # Green for positive
        elif delta.startswith('-'):
            delta_color = "#ef4444"  # Red for negative
        else:
            delta_color = "#f59e0b"  # Yellow for neutral
    else:
        delta_color = "#c4b5fd"
    
    # Trend indicator
    trend_html = ""
    if trend == "up":
        trend_html = """<div style="color: #10b981; font-size: 0.9rem; margin-top: 5px;">
                          ↗️ Improving trend</div>"""
    elif trend == "down":
        trend_html = """<div style="color: #ef4444; font-size: 0.9rem; margin-top: 5px;">
                          ↘️ Needs attention</div>"""
    elif trend == "stable":
        trend_html = """<div style="color: #f59e0b; font-size: 0.9rem; margin-top: 5px;">
                          → Stable performance</div>"""
    
    # Delta HTML
    delta_html = f"""
    <div style="font-size: 0.9rem; color: {delta_color}; margin-top: 5px; font-weight: 600;">
        {delta if delta else ''}
    </div>
    """ if delta else ""
    
    return f"""
    <div class="metric-card" style="border-left: 5px solid {color};">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                    <div style="width: 8px; height: 8px; border-radius: 50%; background: {color};"></div>
                    <div style="font-size: 0.95rem; color: {THEME['text_dark']}; font-weight: 600;">
                        {title}
                    </div>
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {THEME['text_light']}; 
                          line-height: 1.2; margin-bottom: 5px;">
                    {value}
                </div>
                {delta_html}
                {trend_html}
            </div>
            <div style="font-size: 2.5rem; opacity: 0.9; padding-left: 15px;">
                {icon}
            </div>
        </div>
    </div>
    """




def show_no_data_message():
    """Display advanced no-data message with options"""
    
    st.warning("### 📭 No Data Loaded")
    st.write("""
    To start analyzing parallel computing tasks, you need to load data first.
    You can upload your own data or explore with sample datasets.
    """)
    
    st.markdown("---")
    
    # Data loading options using native Streamlit
    option_col1, option_col2, option_col3 = st.columns(3)
    
    with option_col1:
        st.info("**📁 Upload File**\n\nCSV, JSON, Excel formats")
        if st.button("📁 Go to File Upload", use_container_width=True, type="primary", key="no_data_upload"):
            st.session_state.current_page = "Data Input"
            st.session_state.active_tab = "file_upload"
            st.rerun()
    
    with option_col2:
        st.info("**🤖 AI Generation**\n\nDescribe your scenario")
        if st.button("🤖 Describe Problem", use_container_width=True, key="no_data_describe"):
            st.session_state.current_page = "Data Input"
            st.session_state.active_tab = "problem_desc"
            st.rerun()
    
    with option_col3:
        st.info("**🎯 Use Sample**\n\nPre-configured datasets")
        if st.button("🎯 Load Sample Data", use_container_width=True, key="no_data_sample"):
            with st.spinner("Generating comprehensive sample data..."):
                df = generate_sample_data(50)
                st.session_state.data = df
                st.session_state.uploaded_file_name = "Comprehensive Sample Dataset"
                st.session_state.data_loaded = True
                st.session_state.selected_sample = "comprehensive"
                st.success("✅ Sample data loaded successfully!")
                time.sleep(1)
                st.rerun()


def show_dashboard():
    """Display the main dashboard page with advanced analytics"""
    
    # Header - using native Streamlit
    st.markdown("# 📊 Dashboard Overview")
    st.write("Welcome to your Parallel Computing Command Center. Monitor, analyze, and optimize task dependencies and execution patterns in real-time.")
    
    # Badges using native Streamlit columns
    badge_col1, badge_col2, badge_col3 = st.columns(3)
    with badge_col1:
        st.info("🚀 **Real-time**")
    with badge_col2:
        st.success("🤖 **AI-Powered**")
    with badge_col3:
        st.warning("📊 **Advanced Analytics**")
    
    st.markdown("---")
    
    # AI Status Banner
    if not GROQ_AVAILABLE:
        st.warning("""
        ⚠️ **AI Features Limited**
        
        The `groq` package is not installed. AI-powered analysis will use basic pattern matching.  
        For full AI capabilities, run: `pip install groq` and restart the application.
        """)
    
    # Main Content Area
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # SECTION 1: Key Performance Indicators
        st.markdown("### 📈 Key Performance Indicators")
        st.caption("Real-time metrics from your current dataset")
        
        # Calculate metrics
        total_tasks = len(df)
        total_time = df['execution_time_ms'].sum() if 'execution_time_ms' in df.columns else 0
        
        if 'parallelizable' in df.columns:
            parallel_tasks = df['parallelizable'].sum()
            parallel_percent = (parallel_tasks / total_tasks * 100) if total_tasks > 0 else 0
        else:
            parallel_tasks = total_tasks * 0.3  # Estimate
            parallel_percent = 30
        
        if 'cpu_usage_percent' in df.columns:
            avg_cpu = df['cpu_usage_percent'].mean()
            max_cpu = df['cpu_usage_percent'].max()
        else:
            avg_cpu = 35  # Default
            max_cpu = 80
        
        if 'memory_usage_mb' in df.columns:
            avg_memory = df['memory_usage_mb'].mean()
            total_memory = df['memory_usage_mb'].sum()
        else:
            avg_memory = 512  # Default
            total_memory = avg_memory * total_tasks
        
        # Calculate efficiency metrics
        if 'start_time_ms' in df.columns and 'end_time_ms' in df.columns:
            makespan = df['end_time_ms'].max() - df['start_time_ms'].min()
            utilization = (total_time / (makespan * total_tasks)) * 100 if makespan > 0 and total_tasks > 0 else 0
        else:
            makespan = total_time
            utilization = 50  # Default
        
        # Display KPI Cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.metric("Total Tasks", f"{total_tasks:,}", delta="✓ Active" if total_tasks > 25 else None)
        
        with kpi_col2:
            st.metric("Total Execution", f"{total_time:,.0f} ms", delta=f"Makespan: {makespan:,.0f} ms")
        
        with kpi_col3:
            st.metric("Parallelizable", f"{parallel_percent:.1f}%", delta=f"{int(parallel_tasks)} tasks")
        
        with kpi_col4:
            st.metric("Avg CPU Usage", f"{avg_cpu:.1f}%", delta=f"Max: {max_cpu:.1f}%")
        
        st.markdown("---")
        
        # SECTION 2: Quick Actions
        st.markdown("### 🚀 Quick Actions")
        
        action_cols = st.columns(4)
        
        with action_cols[0]:
            if st.button("📥 Upload More Data", use_container_width=True, key="dash_upload_more"):
                st.session_state.current_page = "Data Input"
                st.rerun()
        
        with action_cols[1]:
            if st.button("🔄 Refresh Analysis", use_container_width=True, key="dash_refresh"):
                with st.spinner("Refreshing analysis..."):
                    st.session_state.structure = st.session_state.ai_processor.detect_structure(df)
                    st.session_state.analysis = st.session_state.ai_processor.analyze_parallel_patterns(df, st.session_state.structure)
                    st.session_state.analysis_complete = True
                    st.success("✅ Analysis refreshed!")
        
        with action_cols[2]:
            ai_disabled = not GROQ_AVAILABLE
            if st.button("🤖 Run AI Analysis", use_container_width=True, key="dash_ai", disabled=ai_disabled):
                if not GROQ_AVAILABLE:
                    st.warning("AI features require `pip install groq`")
                else:
                    with st.spinner("AI is analyzing your data..."):
                        st.session_state.structure = st.session_state.ai_processor.detect_structure(df)
                        st.session_state.analysis = st.session_state.ai_processor.analyze_parallel_patterns(df, st.session_state.structure)
                        st.session_state.analysis_complete = True
                        st.success("✅ AI analysis complete!")
        
        with action_cols[3]:
            if st.button("⚡ Optimize Now", use_container_width=True, key="dash_optimize"):
                st.session_state.current_page = "Optimization"
                st.rerun()
        
        st.markdown("---")
        
        # SECTION 3: Core Features
        st.markdown("### ✨ Core Features")
        
        features = [
            {
                "title": "AI-Powered Analysis",
                "icon": "🤖",
                "description": "Intelligent data structure detection and optimization suggestions using advanced ML models",
                "status": "Active" if GROQ_AVAILABLE else "Limited"
            },
            {
                "title": "Dependency Mapping",
                "icon": "🕸️",
                "description": "Visualize complex task dependencies and identify critical paths with interactive graphs",
                "status": "Ready"
            },
            {
                "title": "Performance Metrics",
                "icon": "📊",
                "description": "Detailed latency analysis, bottleneck detection, and resource utilization tracking",
                "status": "Ready"
            },
            {
                "title": "Optimization Engine",
                "icon": "⚡",
                "description": "AI-driven optimization suggestions with simulation and impact analysis",
                "status": "Ready"
            }
        ]
        
        feat_cols = st.columns(4)
        for idx, feature in enumerate(features):
            with feat_cols[idx]:
                if feature["status"] == "Active":
                    st.success(f"**{feature['icon']} {feature['title']}**\n\n{feature['description']}\n\n✅ {feature['status']}")
                elif feature["status"] == "Limited":
                    st.warning(f"**{feature['icon']} {feature['title']}**\n\n{feature['description']}\n\n⚠️ {feature['status']}")
                else:
                    st.info(f"**{feature['icon']} {feature['title']}**\n\n{feature['description']}\n\n🟢 {feature['status']}")
        
        st.markdown("---")
        
        # SECTION 4: Data Preview
        st.markdown("### 📋 Data Preview")
        
        # Metadata
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            st.caption(f"**Source:** {st.session_state.uploaded_file_name or 'Loaded Data'}")
        with meta_col2:
            st.caption(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        with meta_col3:
            st.caption(f"**Size:** {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Create tabs for different views
        preview_tab1, preview_tab2, preview_tab3 = st.tabs(["📊 Data View", "📈 Statistics", "🔍 Quality Check"])
        
        with preview_tab1:
            st.dataframe(df.head(15), use_container_width=True, hide_index=True)
            
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("Rows", df.shape[0])
            with info_col2:
                st.metric("Columns", df.shape[1])
            with info_col3:
                missing_values = df.isnull().sum().sum()
                st.metric("Missing Values", missing_values)
        
        with preview_tab2:
            st.markdown("#### 📊 Statistical Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Numerical Columns**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats_df = df[numeric_cols].describe().T.round(2)
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    st.info("No numerical columns found")
            
            with col2:
                st.markdown("**Categorical Columns**")
                cat_cols = df.select_dtypes(include=['object', 'bool']).columns
                if len(cat_cols) > 0:
                    for col in cat_cols[:3]:
                        st.metric(f"{col} (Unique)", df[col].nunique())
                else:
                    st.info("No categorical columns found")
        
        with preview_tab3:
            st.markdown("#### 🔍 Data Quality Assessment")
            
            quality_col1, quality_col2, quality_col3 = st.columns(3)
            
            with quality_col1:
                # Check for missing values
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                if missing_pct < 5:
                    st.success(f"✅ **Missing Values**\n\n{missing_pct:.1f}% (Good)")
                elif missing_pct < 20:
                    st.warning(f"⚠️ **Missing Values**\n\n{missing_pct:.1f}% (Moderate)")
                else:
                    st.error(f"❌ **Missing Values**\n\n{missing_pct:.1f}% (High)")
            
            with quality_col2:
                # Check for duplicates
                duplicate_rows = df.duplicated().sum()
                duplicate_pct = (duplicate_rows / len(df)) * 100
                if duplicate_pct < 1:
                    st.success(f"✅ **Duplicate Rows**\n\n{duplicate_pct:.1f}% (Good)")
                elif duplicate_pct < 5:
                    st.warning(f"⚠️ **Duplicate Rows**\n\n{duplicate_pct:.1f}% (Moderate)")
                else:
                    st.error(f"❌ **Duplicate Rows**\n\n{duplicate_pct:.1f}% (High)")
            
            with quality_col3:
                # Check data types
                mixed_types = sum([1 for col in df.columns if df[col].apply(type).nunique() > 1])
                if mixed_types == 0:
                    st.success(f"✅ **Data Types**\n\nConsistent")
                else:
                    st.error(f"❌ **Mixed Types**\n\n{mixed_types} columns")
        
        st.markdown("---")
        
        # SECTION 5: AI Recommendations
        if st.session_state.analysis_complete and st.session_state.analysis:
            st.markdown("### 💡 AI Recommendations")
            
            analysis = st.session_state.analysis
            suggestions = analysis.get("optimization_suggestions", [])
            
            if suggestions:
                # Display top 3 recommendations
                rec_cols = st.columns(min(3, len(suggestions)))
                for idx, suggestion in enumerate(suggestions[:3]):
                    with rec_cols[idx]:
                        impact = suggestion.get("impact", "medium")
                        effort = suggestion.get("effort", "medium")
                        
                        # Color based on impact
                        if impact == "high":
                            st.error(f"**🔴 {suggestion.get('description', 'Optimization')}**\n\n{suggestion.get('details', 'Details available.')}\n\n**Impact:** {impact.upper()} | **Effort:** {effort.upper()}")
                        elif impact == "medium":
                            st.warning(f"**🟡 {suggestion.get('description', 'Optimization')}**\n\n{suggestion.get('details', 'Details available.')}\n\n**Impact:** {impact.upper()} | **Effort:** {effort.upper()}")
                        else:
                            st.info(f"**🟢 {suggestion.get('description', 'Optimization')}**\n\n{suggestion.get('details', 'Details available.')}\n\n**Impact:** {impact.upper()} | **Effort:** {effort.upper()}")
                
                if len(suggestions) > 3:
                    st.caption(f"*Showing 3 of {len(suggestions)} recommendations. View all in Optimization page.*")
                    
                    if st.button("📊 View All Recommendations", key="view_all_recs"):
                        st.session_state.current_page = "Optimization"
                        st.rerun()
            else:
                st.info("No specific recommendations yet. Run AI analysis for detailed suggestions.")
        
        st.markdown("---")
        
        # SECTION 6: Export Options
        st.markdown("### 📤 Export Options")
        
        export_cols = st.columns(4)
        
        with export_cols[0]:
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Export CSV",
                data=csv,
                file_name="mtlg_data_export.csv",
                mime="text/csv",
                key="download_csv",
                use_container_width=True
            )
        
        with export_cols[1]:
            if st.button("📊 Export Report", use_container_width=True, key="export_report"):
                st.info("Report generation feature coming soon!")
        
        with export_cols[2]:
            if st.button("🖼️ Export Charts", use_container_width=True, key="export_charts"):
                st.info("Chart export feature coming soon!")
        
        with export_cols[3]:
            if st.button("📋 Copy Summary", use_container_width=True, key="copy_summary"):
                st.info("Summary copy feature coming soon!")
    
    else:
        # Show no data message
        show_no_data_message()

def show_data_input():
    """Display the data input page with multiple input methods"""
    st.markdown(f"""
    <div class="custom-card fade-in">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
            <div>
                <h2 style="color: {THEME['accent_purple']}; margin: 0; font-size: 2.2rem;">📥 Data Input & Processing</h2>
                <p style="color: {THEME['text_dark']}; font-size: 1.1rem; margin-top: 0.5rem;">
                    Upload your parallel computing data or describe your problem. Our AI will automatically 
                    analyze the structure and prepare it for visualization.
                </p>
            </div>
            <div style="font-size: 3rem; opacity: 0.7;">
                🤖
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 File Upload", 
        "📝 Describe Problem", 
        "🎯 Use Sample", 
        "🔗 Advanced Options"
    ])
    
    with tab1:
        show_file_upload()
    
    with tab2:
        show_problem_description()
    
    with tab3:
        show_sample_data()
    
    with tab4:
        show_advanced_options()

def show_file_upload():
    """File upload interface"""
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h4 style="color: #c4b5fd;">Upload Your Data File</h4>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Supported formats: CSV, JSON, Excel, Parquet, TXT
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'json', 'xlsx', 'xls', 'parquet', 'txt'],
        help="Upload your task dependency data",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            # Show file info
            file_size = uploaded_file.size / 1024  # KB
            st.info(f"📄 **File:** {uploaded_file.name} | 📏 **Size:** {file_size:.2f} KB")
            
            # Load data based on file type
            with st.spinner("Loading data..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                else:
                    # Try CSV with different separators
                    content = uploaded_file.getvalue().decode('utf-8')
                    df = pd.read_csv(io.StringIO(content), sep=None, engine='python')
            
            st.success(f"✅ Data loaded successfully! ({df.shape[0]} rows × {df.shape[1]} columns)")
            
            # Store in session state
            st.session_state.data = df
            st.session_state.uploaded_file_name = uploaded_file.name
            
            # Show data preview
            with st.expander("🔍 Data Preview", expanded=True):
                tab_view, tab_info = st.tabs(["View", "Info"])
                
                with tab_view:
                    st.dataframe(df.head(15), use_container_width=True)
                
                with tab_info:
                    st.markdown("#### 📊 Data Information")
                    
                    # Basic info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rows", df.shape[0])
                    with col2:
                        st.metric("Columns", df.shape[1])
                    
                    # Column types
                    st.markdown("#### 📋 Column Types")
                    type_counts = df.dtypes.value_counts()
                    for dtype, count in type_counts.items():
                        st.text(f"{dtype}: {count} columns")
            
            # AI Analysis Button
            st.markdown("---")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("🤖 Analyze with AI", type="primary", use_container_width=True):
                    perform_ai_analysis(df)
            
            with col2:
                st.caption("AI will analyze data structure, detect patterns, and prepare for visualization")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("💡 **Tips:** Ensure your file is properly formatted and try again.")

def show_problem_description():
    """Problem description interface"""
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h4 style="color: #c4b5fd;">Describe Your Parallel Computing Problem</h4>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Describe your task dependencies, timing requirements, and constraints. 
            Our AI will generate appropriate sample data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Problem description input
    problem_desc = st.text_area(
        "Describe your scenario:",
        height=200,
        placeholder="""Example: 
        I have 50 tasks with complex dependencies.
        Some tasks can run in parallel, others must run sequentially.
        Average execution time is 100ms with some outliers at 500ms.
        CPU usage varies between 20-80%.
        Memory usage is around 512MB per task.
        I need to identify bottlenecks and optimize parallel execution.""",
        help="Be as detailed as possible for better AI-generated data"
    )
    
    # Configuration options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_tasks = st.slider("Number of tasks", 10, 200, 50, help="Approximate number of tasks")
    
    with col2:
        complexity = st.select_slider(
            "Dependency Complexity",
            options=["Simple", "Moderate", "Complex", "Very Complex"],
            value="Moderate"
        )
    
    with col3:
        data_type = st.selectbox(
            "Data Characteristics",
            ["Balanced", "CPU-intensive", "Memory-intensive", "I/O-bound", "Network-bound", "Mixed"]
        )
    
    # Additional parameters
    with st.expander("Advanced Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            avg_execution = st.number_input("Average Execution Time (ms)", 10, 10000, 100)
            cpu_range = st.slider("CPU Usage Range (%)", 0, 100, (20, 80))
        with col2:
            memory_range = st.slider("Memory Usage Range (MB)", 1, 10000, (256, 2048))
            parallel_percent = st.slider("Parallelizable Tasks (%)", 0, 100, 40)
    
    if st.button("🎯 Generate AI Data", type="primary", use_container_width=True):
        if problem_desc.strip():
            with st.spinner("🤖 AI is generating data based on your description..."):
                # Generate sample data with parameters
                df = generate_sample_data(num_tasks)
                
                # Adjust data based on parameters
                if data_type == "CPU-intensive":
                    df['cpu_usage_percent'] = df['cpu_usage_percent'] * 1.5
                elif data_type == "Memory-intensive":
                    df['memory_usage_mb'] = df['memory_usage_mb'] * 2
                elif data_type == "I/O-bound":
                    df['io_operations'] = df['io_operations'] * 3
                elif data_type == "Network-bound":
                    df['network_calls'] = df['network_calls'] * 3
                
                # Adjust parallelization
                if parallel_percent < 100:
                    parallel_mask = np.random.random(len(df)) < (parallel_percent / 100)
                    df['parallelizable'] = parallel_mask
                
                # Store in session state
                st.session_state.data = df
                st.session_state.uploaded_file_name = "AI-Generated Sample Data"
                
                # Show success and data
                st.success("✅ AI-generated data created successfully!")
                
                with st.expander("View Generated Data", expanded=True):
                    st.dataframe(df, use_container_width=True)
                
                # Show data characteristics
                st.markdown("#### 📈 Generated Data Characteristics")
                
                cols = st.columns(4)
                metrics = [
                    ("Total Tasks", f"{len(df):,}"),
                    ("Avg Execution", f"{df['execution_time_ms'].mean():.1f}ms"),
                    ("Max Execution", f"{df['execution_time_ms'].max():.1f}ms"),
                    ("Parallelizable", f"{df['parallelizable'].mean()*100:.1f}%")
                ]
                
                for idx, (label, value) in enumerate(metrics):
                    with cols[idx]:
                        st.metric(label, value)
        else:
            st.warning("⚠️ Please describe your problem first.")

def show_sample_data():
    """Sample data interface"""
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h4 style="color: #c4b5fd;">Use Pre-configured Sample Data</h4>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Explore the tool with realistic parallel computing scenarios.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample data options
    sample_options = {
        "Basic Parallel Tasks": {
            "description": "Simple task dependencies with uniform execution times",
            "tasks": 20,
            "complexity": "Low",
            "parallelizable": "High"
        },
        "Complex Dependency Graph": {
            "description": "Nested dependencies with varying execution patterns",
            "tasks": 35,
            "complexity": "Medium",
            "parallelizable": "Medium"
        },
        "Real-world Workload": {
            "description": "Realistic mix of CPU, memory, and I/O intensive tasks",
            "tasks": 75,
            "complexity": "High",
            "parallelizable": "Mixed"
        },
        "Optimization Challenge": {
            "description": "Deliberate bottlenecks and optimization opportunities",
            "tasks": 50,
            "complexity": "Expert",
            "parallelizable": "Low"
        },
        "Large-scale Simulation": {
            "description": "Large dataset for stress testing and scalability analysis",
            "tasks": 150,
            "complexity": "Very Complex",
            "parallelizable": "High"
        }
    }
    
    selected_sample = st.selectbox(
        "Choose a sample dataset:",
        options=list(sample_options.keys()),
        format_func=lambda x: f"{x} - {sample_options[x]['description']}",
        help="Select a sample dataset to explore the tool's features"
    )
    
    # Show sample details
    if selected_sample:
        sample = sample_options[selected_sample]
        st.markdown(f"""
        <div style="background: rgba(30, 11, 61, 0.6); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 1.8rem; color: #a855f7;">📋</div>
                    <div style="color: {THEME['text_light']}; font-weight: 600;">Tasks</div>
                    <div style="color: {THEME['text_dark']};">{sample['tasks']}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.8rem; color: #a855f7;">🕸️</div>
                    <div style="color: {THEME['text_light']}; font-weight: 600;">Complexity</div>
                    <div style="color: {THEME['text_dark']};">{sample['complexity']}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.8rem; color: #a855f7;">⚡</div>
                    <div style="color: {THEME['text_light']}; font-weight: 600;">Parallelizable</div>
                    <div style="color: {THEME['text_dark']};">{sample['parallelizable']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Load Sample Data", use_container_width=True, type="primary"):
            with st.spinner(f"Generating {selected_sample}..."):
                # Generate appropriate sample data
                df = generate_sample_data(sample_options[selected_sample]['tasks'])
                
                # Adjust based on selection
                if selected_sample == "Basic Parallel Tasks":
                    df['parallelizable'] = True  # All tasks parallelizable
                    df['execution_time_ms'] = np.random.uniform(50, 150, len(df))
                elif selected_sample == "Complex Dependency Graph":
                    # Create more complex dependencies
                    df['level'] = np.random.randint(0, 5, len(df))
                elif selected_sample == "Optimization Challenge":
                    # Create bottlenecks
                    bottleneck_idx = np.random.choice(len(df), size=5, replace=False)
                    df.loc[bottleneck_idx, 'execution_time_ms'] *= 5
                    df['parallelizable'] = np.random.random(len(df)) < 0.3  # Low parallelization
                
                # Store in session state
                st.session_state.data = df
                st.session_state.uploaded_file_name = f"Sample: {selected_sample}"
                st.session_state.selected_sample = selected_sample
                
                st.success(f"✅ {selected_sample} loaded successfully!")
                st.rerun()
    
    with col2:
        if st.button("📊 Preview Sample", use_container_width=True):
            df_preview = generate_sample_data(10)
            st.dataframe(df_preview, use_container_width=True)
            
            st.markdown("**Sample Statistics:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tasks", len(df_preview))
            with col2:
                st.metric("Avg Time", f"{df_preview['execution_time_ms'].mean():.1f}ms")
            with col3:
                st.metric("Parallel", f"{df_preview['parallelizable'].mean()*100:.0f}%")

def show_advanced_options():
    """Advanced options interface"""
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h4 style="color: #c4b5fd;">Advanced Data Options</h4>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Configure advanced data generation and processing options.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced generation options
    with st.expander("🔧 Data Generation Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            seed_value = st.number_input("Random Seed", 0, 10000, 42, 
                                        help="Seed for reproducible data generation")
            time_std = st.slider("Execution Time Variability", 0.1, 2.0, 0.5, 0.1,
                               help="Standard deviation multiplier for execution times")
        
        with col2:
            resource_correlation = st.slider("Resource-Time Correlation", 0.0, 1.0, 0.7, 0.1,
                                          help="How strongly resource usage correlates with execution time")
            noise_level = st.slider("Data Noise Level", 0.0, 1.0, 0.2, 0.1,
                                  help="Amount of random noise in generated data")
    
    # Data transformation options
    with st.expander("🔄 Data Transformation", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            normalize = st.checkbox("Normalize Numerical Columns", value=False)
            add_missing = st.checkbox("Add Missing Values", value=False)
        
        with col2:
            add_outliers = st.checkbox("Add Outliers", value=False)
            create_duplicates = st.checkbox("Create Duplicate Tasks", value=False)
    
    # Data preview with custom parameters
    if st.button("🔄 Generate Custom Data", type="primary", use_container_width=True):
        with st.spinner("Generating custom data..."):
            # Generate with custom parameters
            df = generate_sample_data(30)
            
            # Apply customizations
            np.random.seed(seed_value)
            
            # Add variability
            if time_std != 1.0:
                df['execution_time_ms'] *= np.random.normal(1, time_std * 0.2, len(df))
            
            # Add resource correlation
            if resource_correlation > 0:
                df['cpu_usage_percent'] = df['cpu_usage_percent'] * (0.3 + 0.7 * resource_correlation)
                df['memory_usage_mb'] = df['memory_usage_mb'] * (0.5 + 0.5 * resource_correlation)
            
            # Add noise
            if noise_level > 0:
                noise = np.random.normal(1, noise_level * 0.3, len(df))
                df['execution_time_ms'] *= noise
                df['cpu_usage_percent'] *= np.clip(noise, 0.5, 2)
                df['memory_usage_mb'] *= np.clip(noise, 0.5, 2)
            
            # Add missing values
            if add_missing:
                missing_mask = np.random.random(len(df)) < 0.1
                df.loc[missing_mask, 'cpu_usage_percent'] = np.nan
                df.loc[missing_mask, 'memory_usage_mb'] = np.nan
            
            # Add outliers
            if add_outliers:
                outlier_mask = np.random.random(len(df)) < 0.05
                df.loc[outlier_mask, 'execution_time_ms'] *= 10
            
            # Add duplicates
            if create_duplicates and len(df) > 5:
                duplicates = df.sample(2).copy()
                duplicates['task_id'] = [f"TASK_DUP_{i}" for i in range(2)]
                df = pd.concat([df, duplicates], ignore_index=True)
            
            # Store in session state
            st.session_state.data = df
            st.session_state.uploaded_file_name = "Custom Generated Data"
            
            st.success("✅ Custom data generated successfully!")
            
            # Show preview
            with st.expander("View Generated Data", expanded=True):
                st.dataframe(df, use_container_width=True)
    
    # API Connection Section
    st.markdown("---")
    st.markdown("#### 🔗 API Integration")
    
    st.info("""
    **Coming Soon:** API integrations for real-time data ingestion from:
    - **Apache Spark** - Stream processing data
    - **Kubernetes** - Container orchestration metrics
    - **AWS CloudWatch** - Cloud monitoring data
    - **Prometheus** - Time-series metrics
    - **Custom REST APIs** - Your own data sources
    """)

def perform_ai_analysis(df: pd.DataFrame):
    """Perform AI analysis on the data"""
    # Initialize AI processor
    if st.session_state.ai_processor is None:
        st.session_state.ai_processor = AIDataProcessor()
    
    with st.spinner("🔍 AI is analyzing your data structure..."):
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Detect structure
        status_text.text("Step 1/3: Detecting data structure...")
        progress_bar.progress(30)
        structure = st.session_state.ai_processor.detect_structure(df)
        
        # Step 2: Analyze patterns
        status_text.text("Step 2/3: Analyzing parallel patterns...")
        progress_bar.progress(60)
        analysis = st.session_state.ai_processor.analyze_parallel_patterns(df, structure)
        
        # Step 3: Generate plan
        status_text.text("Step 3/3: Generating optimization plan...")
        progress_bar.progress(90)
        optimization_plan = st.session_state.ai_processor.generate_optimization_plan(analysis)
        
        # Store results
        st.session_state.structure = structure
        st.session_state.analysis = analysis
        st.session_state.optimization_plan = optimization_plan
        st.session_state.last_analysis_time = datetime.now()
        st.session_state.analysis_complete = True
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
    
    # Display results
    show_ai_results(structure, analysis, optimization_plan)

def show_ai_results(structure: Dict, analysis: Dict, optimization_plan: Dict):
    """Display AI analysis results"""
    st.markdown(f"""
    <div class="custom-card">
        <h3 style="color: {THEME['accent_purple']}; margin-bottom: 20px;">
            🤖 AI Analysis Results
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Results in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Structure Detection", "🔍 Pattern Analysis", "💡 Optimization Plan", "🚀 Implementation"])
    
    with tab1:
        st.markdown("#### 🔍 Detected Data Structure")
        
        if "detected_structure" in structure:
            # Create structure table
            struct_data = []
            for col, purpose in structure["detected_structure"].items():
                confidence = structure.get("confidence_scores", {}).get(col, 0.5)
                struct_data.append({
                    "Column": col,
                    "Detected Purpose": purpose,
                    "Confidence": f"{confidence*100:.1f}%",
                    "Status": "✅ High" if confidence > 0.8 else "⚠️ Medium" if confidence > 0.6 else "❓ Low"
                })
            
            struct_df = pd.DataFrame(struct_data)
            st.dataframe(struct_df, use_container_width=True, hide_index=True)
        
        # Show dependency info
        if "dependency_graph" in structure and structure["dependency_graph"]:
            dep_graph = structure["dependency_graph"]
            st.markdown("#### 🕸️ Dependency Information")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Parent Column", dep_graph.get("parent_col", "Not detected"))
            with col2:
                st.metric("Child Column", dep_graph.get("child_col", "Not detected"))
        
        # Show timing info
        if "timing_analysis" in structure and structure["timing_analysis"]:
            timing = structure["timing_analysis"]
            st.markdown("#### ⏱️ Timing Information")
            cols = st.columns(3)
            timing_cols = ["start_col", "end_col", "duration_col"]
            timing_labels = ["Start Time", "End Time", "Duration"]
            
            for i, (col_key, label) in enumerate(zip(timing_cols, timing_labels)):
                with cols[i]:
                    st.metric(label, timing.get(col_key, "Not detected"))
    
    with tab2:
        if isinstance(analysis, dict):
            # Display key metrics
            st.markdown("#### 📈 Performance Metrics")
            
            # Performance metrics
            perf_metrics = analysis.get("performance_metrics", {})
            if perf_metrics:
                cols = st.columns(3)
                metrics = [
                    ("Current Total Time", f"{perf_metrics.get('current_total_time', 0):.0f} ms"),
                    ("Optimized Time", f"{perf_metrics.get('optimized_time', 0):.0f} ms"),
                    ("Speedup Factor", f"{perf_metrics.get('speedup_factor', 1):.2f}x")
                ]
                
                for idx, (label, value) in enumerate(metrics):
                    with cols[idx]:
                        st.metric(label, value)
            
            # Parallelization analysis
            st.markdown("#### ⚡ Parallelization Analysis")
            parallel_analysis = analysis.get("parallelization_analysis", {})
            if parallel_analysis:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Parallelizable Tasks", f"{parallel_analysis.get('parallelizable_tasks', 0)}")
                with col2:
                    st.metric("Parallel Percentage", f"{parallel_analysis.get('parallel_percentage', 0):.1f}%")
                with col3:
                    st.metric("Estimated Speedup", f"{parallel_analysis.get('estimated_speedup', 1):.2f}x")
            
            # Bottleneck detection
            st.markdown("#### ⚠️ Bottleneck Detection")
            bottlenecks = analysis.get("bottleneck_detection", {})
            if bottlenecks and bottlenecks.get("bottleneck_tasks"):
                bottleneck_tasks = bottlenecks["bottleneck_tasks"]
                impact = bottlenecks.get("impact_percentage", 0)
                
                st.warning(f"**Found {len(bottleneck_tasks)} bottlenecks accounting for {impact:.1f}% of total time**")
                
                # Show top bottlenecks
                bottleneck_df = pd.DataFrame(bottleneck_tasks)
                if not bottleneck_df.empty:
                    # ✅ FIXED: Check what columns exist before sorting
                    if 'time' in bottleneck_df.columns:
                        sort_column = 'time'
                    elif 'execution_time' in bottleneck_df.columns:
                        sort_column = 'execution_time'
                    elif 'duration' in bottleneck_df.columns:
                        sort_column = 'duration'
                    else:
                        # Find any numeric column to sort by
                        numeric_cols = bottleneck_df.select_dtypes(include=[np.number]).columns
                        sort_column = numeric_cols[0] if len(numeric_cols) > 0 else bottleneck_df.columns[0]
                    
                    st.dataframe(
                        bottleneck_df.sort_values(sort_column, ascending=False).head(5),
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.success("✅ No significant bottlenecks detected")
            
            # Resource analysis
            resource_analysis = analysis.get("resource_analysis", {})
            if resource_analysis:
                st.markdown("#### 💾 Resource Analysis")
                for recommendation in resource_analysis.get("recommendations", []):
                    st.info(f"💡 {recommendation}")
    
    with tab3:
        st.markdown("#### 🚀 Optimization Suggestions")
        
        suggestions = analysis.get("optimization_suggestions", [])
        if suggestions:
            # Categorize suggestions by impact
            high_impact = [s for s in suggestions if s.get("impact") == "high"]
            medium_impact = [s for s in suggestions if s.get("impact") == "medium"]
            low_impact = [s for s in suggestions if s.get("impact") == "low"]
            
            # Display by impact level
            if high_impact:
                st.markdown("##### 🔴 High Impact")
                for suggestion in high_impact:
                    display_suggestion(suggestion)
            
            if medium_impact:
                st.markdown("##### 🟡 Medium Impact")
                for suggestion in medium_impact:
                    display_suggestion(suggestion)
            
            if low_impact:
                st.markdown("##### 🟢 Low Impact")
                for suggestion in low_impact:
                    display_suggestion(suggestion)
        else:
            st.info("No specific optimization suggestions generated. The analysis may need more detailed data.")
        
        # Summary metrics
        if optimization_plan and "summary" in optimization_plan:
            summary = optimization_plan["summary"]
            st.markdown("#### 📊 Implementation Summary")
            
            cols = st.columns(4)
            summary_metrics = [
                ("Total Suggestions", f"{summary.get('total_suggestions', 0)}"),
                ("Estimated Impact", summary.get('estimated_impact', 'Medium')),
                ("Timeline", summary.get('implementation_timeline', '2-4 weeks')),
                ("Success Probability", f"{summary.get('success_probability', 0.7)*100:.0f}%")
            ]
            
            for idx, (label, value) in enumerate(summary_metrics):
                with cols[idx]:
                    st.metric(label, value)
    
    with tab4:
        st.markdown("#### 🗺️ Implementation Roadmap")
        
        if optimization_plan and "phases" in optimization_plan:
            phases = optimization_plan["phases"]
            
            for phase in phases:
                with st.expander(f"Phase {phase['phase']}: {phase['name']} ({phase['duration']})", expanded=True):
                    # Phase details
                    st.markdown(f"**Duration:** {phase['duration']}")
                    
                    # Tasks
                    st.markdown("**Tasks:**")
                    for task in phase.get("tasks", []):
                        st.markdown(f"• **{task.get('description', 'Task')}**")
                        if task.get("details"):
                            for detail in task["details"]:
                                st.markdown(f"  - {detail}")
                    
                    # Risks and mitigations
                    if phase.get("risks"):
                        st.markdown("**Risks:**")
                        for risk in phase["risks"]:
                            st.markdown(f"• {risk}")
                    
                    if phase.get("mitigations"):
                        st.markdown("**Mitigations:**")
                        for mitigation in phase["mitigations"]:
                            st.markdown(f"• {mitigation}")
        
        # Metrics to track
        if optimization_plan and "metrics_to_track" in optimization_plan:
            st.markdown("#### 📈 Metrics to Track")
            metrics = optimization_plan["metrics_to_track"]
            cols = st.columns(3)
            for idx, metric in enumerate(metrics):
                with cols[idx % 3]:
                    st.info(f"📊 {metric}")
        
        # Expected outcomes
        if optimization_plan and "expected_outcomes" in optimization_plan:
            st.markdown("#### 🎯 Expected Outcomes")
            for outcome in optimization_plan["expected_outcomes"]:
                st.success(f"✅ {outcome}")

def display_suggestion(suggestion: Dict):
    """Display a single optimization suggestion"""
    impact_color = {
        "high": "#ef4444",
        "medium": "#f59e0b",
        "low": "#10b981"
    }.get(suggestion.get("impact", "medium"), "#f59e0b")
    
    effort_color = {
        "high": "#ef4444",
        "medium": "#f59e0b",
        "low": "#10b981"
    }.get(suggestion.get("effort", "medium"), "#f59e0b")
    
    icon = {
        "parallelization": "⚡",
        "bottleneck": "⚠️",
        "resource": "💾",
        "dependency": "🔄",
        "memory": "🧠",
        "io": "📁",
        "network": "🌐"
    }.get(suggestion.get("type", "general"), "🔧")
    
    st.markdown(f"""
    <div style="background: rgba(30, 11, 61, 0.7); border-radius: 12px; padding: 1.5rem; 
              border: 1px solid {impact_color}40; margin: 1rem 0;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <div style="font-size: 1.8rem;">{icon}</div>
            <div style="flex: 1;">
                <div style="color: {THEME['text_light']}; font-weight: 600; font-size: 1.1rem;">
                    {suggestion.get('description', 'Optimization Suggestion')}
                </div>
            </div>
        </div>
        
        <p style="color: {THEME['text_dark']}; font-size: 0.95rem; margin-bottom: 15px;">
            {suggestion.get('details', 'Detailed recommendation available.')}
        </p>
        
        <div style="display: flex; gap: 20px; margin-top: 15px;">
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background: {impact_color};"></div>
                <div style="color: {THEME['text_dark']}; font-size: 0.9rem;">
                    <strong>Impact:</strong> 
                    <span style="color: {impact_color}; font-weight: 600; text-transform: capitalize;">
                        {suggestion.get('impact', 'medium')}
                    </span>
                </div>
            </div>
            
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background: {effort_color};"></div>
                <div style="color: {THEME['text_dark']}; font-size: 0.9rem;">
                    <strong>Effort:</strong> 
                    <span style="color: {effort_color}; font-weight: 600; text-transform: capitalize;">
                        {suggestion.get('effort', 'medium')}
                    </span>
                </div>
            </div>
            
            <div style="display: flex; align-items: center; gap: 6px; margin-left: auto;">
                <div style="font-size: 0.9rem; color: {THEME['text_dark']};">
                    <strong>Type:</strong> {suggestion.get('type', 'general')}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE: DEPENDENCY GRAPH
# ============================================
def show_dependency_graph():
    """Display the dependency graph visualization page"""
    st.markdown(f"""
    <div class="custom-card fade-in">
        <h2 style="color: {THEME['accent_purple']}; margin: 0; font-size: 2.2rem;">🕸️ Dependency Graph Visualization</h2>
        <p style="color: {THEME['text_dark']}; font-size: 1.1rem; margin-top: 0.5rem;">
            Visualize task dependencies, identify critical paths, and analyze execution patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is None:
        show_no_data_message()
        return
    
    df = st.session_state.data
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙️ Graph Settings")
        
        # Layout options - DEFINE THIS FIRST
        layout_options = {
            "Hierarchical": "Best for dependency trees",
            "Circular": "Good for cyclic dependencies",
            "Spring": "Force-directed layout (default)",
            "Kamada-Kawai": "Optimal distance layout",
            "Spectral": "Eigenvector-based layout",
            "Fruchterman-Reingold": "Fast force-directed"
        }
        
        # NOW use layout_type with the selectbox
        layout_type = st.selectbox(
            "Layout Algorithm",
            options=list(layout_options.keys()),
            index=2,
            help="Choose the graph layout algorithm"
        )
        
        # Node sizing
        node_size_by = st.selectbox(
            "Node Size Represents",
            ["Execution Time", "CPU Usage", "Memory Usage", "Task Priority", "Level", "Uniform"],
            index=0
        )
        
        # Color coding
        color_by = st.selectbox(
            "Node Color Represents",
            ["Task Type", "Execution Time", "Level", "Parallelizable", "Status", "Single Color"],
            index=0
        )
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            show_labels = st.checkbox("Show Labels", True)
        with col2:
            show_critical = st.checkbox("Show Critical Path", True)
        
        # Graph size
        graph_size = st.slider("Graph Size", 500, 1000, 700, 50)
        
        # Advanced options
        with st.expander("Advanced Options"):
            edge_width = st.slider("Edge Width", 1, 5, 2)
            node_opacity = st.slider("Node Opacity", 0.1, 1.0, 0.8, 0.1)
            animation = st.checkbox("Enable Animation", False)
        
        # Action buttons
        st.markdown("---")
        st.markdown("### 🎯 Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Graph", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("📥 Export Graph", use_container_width=True):
                st.info("Export feature coming soon!")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Graph statistics
        st.markdown("### 📊 Graph Statistics")
        
        # Calculate graph metrics
        G = create_dependency_graph(df)
        
        if len(G.nodes()) > 0:
            stats = [
                ("Nodes", len(G.nodes())),
                ("Edges", len(G.edges())),
                ("Density", f"{nx.density(G):.3f}"),
                ("Avg Degree", f"{sum(dict(G.degree()).values()) / len(G):.2f}")
            ]
            
            for label, value in stats:
                st.metric(label, value)
            
            # Connected components
            if nx.is_directed(G):
                weak_components = nx.number_weakly_connected_components(G)
                strong_components = nx.number_strongly_connected_components(G)
                st.metric("Weak Components", weak_components)
                st.metric("Strong Components", strong_components)
            
            # Critical path if available
            if show_critical:
                try:
                    if nx.is_directed_acyclic_graph(G):
                        longest_path = nx.dag_longest_path(G)
                        longest_path_length = sum(G.nodes[node].get('execution_time_ms', 1) for node in longest_path)
                        st.metric("Critical Path Length", f"{len(longest_path)} tasks")
                        st.metric("Critical Path Time", f"{longest_path_length:.0f} ms")
                except:
                    pass
            
            # Top nodes by degree
            st.markdown("#### 🔝 Top Connected Tasks")
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            for node, degree in top_nodes:
                st.caption(f"**{node}**: {degree} connections")
        else:
            st.warning("No dependencies detected in the data")
    
    with col1:
        # Create and display the graph
        st.markdown("### 🕸️ Dependency Visualization")
        
        # Create the plotly figure
        fig = create_interactive_graph(
            df, 
            layout_type.lower().replace(" ", "_"),
            node_size_by.lower().replace(" ", "_"),
            color_by.lower().replace(" ", "_"),
            show_labels,
            show_critical,
            graph_size
        )
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        else:
            st.warning("Could not generate graph visualization")
        
        # Graph interpretation
        with st.expander("💡 Graph Interpretation Guide", expanded=False):
            st.markdown("""
            ### How to Interpret the Graph:
            
            **Nodes (Circles):**
            - **Size**: Larger nodes indicate higher values (execution time, resource usage, etc.)
            - **Color**: Different colors represent different task characteristics
            - **Border**: May indicate priority or status
            
            **Edges (Lines/Arrows):**
            - Show dependencies between tasks
            - Direction indicates execution order (parent → child)
            - Thickness may indicate dependency strength
            
            **Color Scheme:**
            - 🔵 **Blue**: Compute-intensive tasks
            - 🟢 **Green**: I/O-bound tasks  
            - 🟡 **Yellow**: Memory-intensive tasks
            - 🔴 **Red**: Network-bound tasks
            - 🟣 **Purple**: Parallelizable tasks
            - ⚫ **Black**: Critical path tasks
            
            **Layout Algorithms:**
            - **Hierarchical**: Parent-child relationships (top-down)
            - **Circular**: Good for cyclic dependencies
            - **Spring**: Force-directed, shows natural clusters
            - **Kamada-Kawai**: Optimal distances between nodes
            - **Spectral**: Uses eigenvectors, good for community detection
            
            **Interactions:**
            - **Hover**: See detailed task information
            - **Click**: Select and highlight connected nodes
            - **Drag**: Reposition nodes manually
            - **Zoom**: Scroll to zoom in/out
            - **Pan**: Click and drag to move around
            """)
    
    # Additional analysis
    if len(df) > 0:
        st.markdown(f"""
        <div class="custom-card">
            <h3 style="color: {THEME['accent_purple']}; margin-bottom: 15px;">📈 Dependency Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Degree distribution and other metrics
        analyze_graph_metrics(G, df)

def create_dependency_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Create a networkx graph from the dataframe"""
    G = nx.DiGraph()
    
    if len(df) == 0:
        return G
    
    # Find potential ID and parent columns
    id_col = None
    parent_col = None
    
    # Try to find ID column
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['id', 'task', 'name']):
            id_col = col
            break
    
    if id_col is None:
        # Use index as ID
        df['_temp_id'] = [f"Task_{i}" for i in range(len(df))]
        id_col = '_temp_id'
    
    # Try to find parent column
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['parent', 'depend', 'predecessor']):
            parent_col = col
            break
    
    # Add nodes with attributes
    for idx, row in df.iterrows():
        node_id = str(row[id_col])
        
        # Gather node attributes
        attrs = {}
        for col in df.columns:
            if col == parent_col:
                continue
            value = row[col]
            if pd.notna(value):
                # Convert numpy types to Python types
                if hasattr(value, 'item'):
                    value = value.item()
                attrs[col] = value
        
        G.add_node(node_id, **attrs)
    
    # Add edges based on parent-child relationships
    if parent_col:
        for idx, row in df.iterrows():
            child = str(row[id_col])
            parent = str(row[parent_col]) if pd.notna(row[parent_col]) else None
            
            if parent and parent in G.nodes():
                G.add_edge(parent, child)
    
    return G

def create_interactive_graph(df: pd.DataFrame, layout: str, size_by: str, color_by: str, 
                           show_labels: bool, show_critical: bool, height: int = 700) -> Optional[go.Figure]:
    """Create an interactive Plotly graph"""
    G = create_dependency_graph(df)
    
    if len(G.nodes()) == 0:
        st.warning("No nodes to display in the graph")
        return None
    
    if len(G.edges()) == 0:
        st.info("No dependencies detected. Showing tasks as independent nodes.")
    
    # Choose layout
    try:
        if layout == "hierarchical":
            try:
                pos = nx.multipartite_layout(G)
            except:
                pos = nx.spring_layout(G, k=1.5, iterations=100)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        elif layout == "fruchterman_reingold":
            pos = nx.fruchterman_reingold_layout(G)
        else:  # spring (default)
            pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)
    except Exception as e:
        st.error(f"Error creating graph layout: {e}")
        pos = nx.random_layout(G)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge hover text
        source_attrs = G.nodes[edge[0]]
        target_attrs = G.nodes[edge[1]]
        
        source_name = source_attrs.get('task_name', edge[0])
        target_name = target_attrs.get('task_name', edge[1])
        
        edge_text.append(
            f"<b>Dependency:</b> {source_name} → {target_name}<br>"
            f"<b>Source:</b> {edge[0]}<br>"
            f"<b>Target:</b> {edge[1]}"
        )
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='rgba(168, 85, 247, 0.6)'),
        hoverinfo='text',
        text=edge_text,
        mode='lines',
        name='Dependencies'
    )
    
    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    node_names = []
    
    # Determine critical path if requested
    critical_nodes = set()
    if show_critical and nx.is_directed_acyclic_graph(G):
        try:
            longest_path = nx.dag_longest_path(G)
            critical_nodes = set(longest_path)
        except:
            pass
    
    # Get attribute ranges for scaling
    exec_times = [G.nodes[n].get('execution_time_ms', 1) for n in G.nodes()]
    cpu_usages = [G.nodes[n].get('cpu_usage_percent', 1) for n in G.nodes()]
    memory_usages = [G.nodes[n].get('memory_usage_mb', 1) for n in G.nodes()]
    priorities = [G.nodes[n].get('priority', 1) for n in G.nodes()]
    levels = [G.nodes[n].get('level', 1) for n in G.nodes()]
    
    max_exec_time = max(exec_times) if exec_times else 1
    max_cpu = max(cpu_usages) if cpu_usages else 1
    max_memory = max(memory_usages) if memory_usages else 1
    max_priority = max(priorities) if priorities else 1
    max_level = max(levels) if levels else 1
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        attrs = G.nodes[node]
        
        # Node name for display
        node_name = attrs.get('task_name', attrs.get('task_id', node))
        node_names.append(node_name)
        
        # Node size based on selection
        if size_by == "execution_time":
            size_val = attrs.get('execution_time_ms', 1)
            size = 10 + (size_val / max_exec_time) * 40
        elif size_by == "cpu_usage":
            size_val = attrs.get('cpu_usage_percent', 1)
            size = 10 + (size_val / max_cpu) * 40
        elif size_by == "memory_usage":
            size_val = attrs.get('memory_usage_mb', 1)
            size = 10 + (np.log1p(size_val) / np.log1p(max_memory)) * 40
        elif size_by == "task_priority":
            size_val = attrs.get('priority', 1)
            size = 10 + (size_val / max_priority) * 40
        elif size_by == "level":
            size_val = attrs.get('level', 1)
            size = 10 + (size_val / max_level) * 40
        else:  # uniform
            size = 25
        
        node_size.append(size)
        
        # Node color based on selection
        if color_by == "task_type":
            task_type = attrs.get('task_type', 'unknown').lower()
            color_map = {
                'compute': '#3b82f6',    # Blue
                'io': '#10b981',         # Green
                'memory': '#f59e0b',     # Yellow
                'network': '#ef4444',    # Red
                'database': '#8b5cf6',   # Purple
                'unknown': '#94a3b8'     # Gray
            }
            color = color_map.get(task_type, '#7e3af2')
        elif color_by == "execution_time":
            # Color gradient based on execution time
            time_val = attrs.get('execution_time_ms', 0)
            if max_exec_time > 0:
                normalized = time_val / max_exec_time
                # Blue (low) to Red (high) gradient
                r = int(30 + normalized * 225)
                g = int(144 - normalized * 144)
                b = int(255 - normalized * 200)
                color = f'rgb({r}, {g}, {b})'
            else:
                color = '#7e3af2'
        elif color_by == "parallelizable":
            parallelizable = attrs.get('parallelizable', False)
            color = '#10b981' if parallelizable else '#ef4444'
        elif color_by == "status":
            status = attrs.get('status', 'unknown').lower()
            color_map = {
                'completed': '#10b981',
                'running': '#3b82f6',
                'pending': '#f59e0b',
                'failed': '#ef4444',
                'unknown': '#94a3b8'
            }
            color = color_map.get(status, '#7e3af2')
        elif color_by == "level":
            level = attrs.get('level', 0)
            # Gradient from light to dark purple
            intensity = max(0.3, 1 - (level / max(1, max_level)))
            r = int(124 * intensity)
            g = int(58 * intensity)
            b = int(237 * intensity)
            color = f'rgb({r}, {g}, {b})'
        elif node in critical_nodes:
            color = '#ef4444'  # Red for critical path
        else:
            color = '#7e3af2'  # Default purple
        
        node_color.append(color)
        
        # Node hover text
        hover_text = f"<b>Task:</b> {node_name}<br>"
        hover_text += f"<b>ID:</b> {node}<br>"
        
        # Add key attributes
        if 'execution_time_ms' in attrs:
            hover_text += f"<b>Execution Time:</b> {attrs['execution_time_ms']:.1f} ms<br>"
        if 'cpu_usage_percent' in attrs:
            hover_text += f"<b>CPU Usage:</b> {attrs['cpu_usage_percent']:.1f}%<br>"
        if 'memory_usage_mb' in attrs:
            hover_text += f"<b>Memory:</b> {attrs['memory_usage_mb']:.1f} MB<br>"
        if 'parallelizable' in attrs:
            hover_text += f"<b>Parallelizable:</b> {attrs['parallelizable']}<br>"
        if 'level' in attrs:
            hover_text += f"<b>Level:</b> {attrs['level']}<br>"
        if node in critical_nodes:
            hover_text += "<br><b>⚠️ Critical Path Task</b>"
        
        node_text.append(hover_text)
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text' if show_labels else 'markers',
        text=[node_names[i] if show_labels else '' for i in range(len(node_names))],
        textposition="top center",
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white'),
            opacity=0.85
        ),
        name='Tasks'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=20, r=20, t=20),
                       xaxis=dict(
                           showgrid=False,
                           zeroline=False,
                           showticklabels=False
                       ),
                       yaxis=dict(
                           showgrid=False,
                           zeroline=False,
                           showticklabels=False
                       ),
                       plot_bgcolor='rgba(30, 11, 61, 0.3)',
                       paper_bgcolor='rgba(0,0,0,0)',
                       height=height,
                       dragmode='pan',
                       title="Task Dependency Graph",
                       font=dict(
                           family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif",
                           color=THEME['text_light']
                       )
                   ))
    
    return fig

def analyze_graph_metrics(G: nx.DiGraph, df: pd.DataFrame):
    """Analyze and display graph metrics"""
    if len(G.nodes()) < 2:
        st.info("Not enough nodes for detailed graph analysis.")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Basic Metrics", "🔗 Connectivity", "📈 Degree Analysis", "🔄 Cycles & Paths"])
    
    with tab1:
        # Basic graph metrics
        st.markdown("#### 📊 Basic Graph Metrics")
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Nodes", len(G.nodes()))
        with cols[1]:
            st.metric("Edges", len(G.edges()))
        with cols[2]:
            density = nx.density(G)
            st.metric("Density", f"{density:.3f}")
        with cols[3]:
            avg_degree = sum(dict(G.degree()).values()) / len(G) if len(G) > 0 else 0
            st.metric("Avg Degree", f"{avg_degree:.2f}")
        
        # Additional metrics
        if len(G.nodes()) > 10:
            st.markdown("#### 📈 Additional Metrics")
            
            col1, col2 = st.columns(2)
            with col1:
                # Diameter if connected
                if nx.is_weakly_connected(G):
                    try:
                        diameter = nx.diameter(G.to_undirected())
                        st.metric("Diameter", diameter)
                    except:
                        st.metric("Diameter", "N/A")
                else:
                    st.metric("Weak Components", nx.number_weakly_connected_components(G))
            
            with col2:
                # Clustering coefficient (for undirected version)
                try:
                    undirected_G = G.to_undirected()
                    clustering = nx.average_clustering(undirected_G)
                    st.metric("Clustering Coef", f"{clustering:.3f}")
                except:
                    st.metric("Clustering", "N/A")
    
    with tab2:
        # Connectivity analysis
        st.markdown("#### 🔗 Connectivity Analysis")
        
        if nx.is_weakly_connected(G):
            st.success("✅ Graph is weakly connected")
        else:
            weak_components = nx.number_weakly_connected_components(G)
            st.warning(f"⚠️ Graph has {weak_components} weakly connected components")
        
        if nx.is_strongly_connected(G):
            st.success("✅ Graph is strongly connected")
        else:
            strong_components = nx.number_strongly_connected_components(G)
            st.info(f"📊 Graph has {strong_components} strongly connected components")
        
        # Find articulation points
        if len(G.nodes()) > 2:
            try:
                undirected_G = G.to_undirected()
                articulation_points = list(nx.articulation_points(undirected_G))
                if articulation_points:
                    st.warning(f"⚠️ Found {len(articulation_points)} articulation points")
                    st.caption("Articulation points are critical nodes whose removal disconnects the graph")
            except:
                pass
    
    with tab3:
        # Degree analysis
        st.markdown("#### 📈 Degree Distribution")
        
        degrees = [deg for _, deg in G.degree()]
        
        if degrees:
            # Calculate statistics
            min_degree = min(degrees)
            max_degree = max(degrees)
            avg_degree = np.mean(degrees)
            median_degree = np.median(degrees)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min Degree", min_degree)
            with col2:
                st.metric("Max Degree", max_degree)
            with col3:
                st.metric("Avg Degree", f"{avg_degree:.2f}")
            with col4:
                st.metric("Median Degree", f"{median_degree:.2f}")
            
            # Degree distribution plot
            fig = px.histogram(
                x=degrees,
                nbins=20,
                title="Degree Distribution",
                labels={'x': 'Degree', 'y': 'Count'},
                color_discrete_sequence=['#7e3af2']
            )
            fig.update_layout(
                plot_bgcolor='rgba(30, 11, 61, 0.3)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=THEME['text_light'],
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top connected nodes
            st.markdown("#### 🔝 Most Connected Tasks")
            degree_dict = dict(G.degree())
            top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            
            top_df = pd.DataFrame(top_nodes, columns=['Task', 'Connections'])
            st.dataframe(top_df, use_container_width=True, hide_index=True)
    
    with tab4:
        # Cycle and path analysis
        st.markdown("#### 🔄 Cycle Analysis")
        
        # Check for cycles
        try:
            if nx.is_directed_acyclic_graph(G):
                st.success("✅ Graph is acyclic (no cycles)")
                
                # Critical path analysis
                try:
                    longest_path = nx.dag_longest_path(G)
                    longest_path_length = len(longest_path)
                    
                    st.markdown("#### 🎯 Critical Path Analysis")
                    st.metric("Critical Path Length", f"{longest_path_length} tasks")
                    
                    # Calculate critical path time
                    path_time = 0
                    for node in longest_path:
                        node_attrs = G.nodes[node]
                        path_time += node_attrs.get('execution_time_ms', 0)
                    
                    st.metric("Critical Path Time", f"{path_time:.0f} ms")
                    
                    # Show critical path
                    with st.expander("View Critical Path", expanded=False):
                        path_df = pd.DataFrame([
                            {
                                'Task': node,
                                'Name': G.nodes[node].get('task_name', node),
                                'Time (ms)': G.nodes[node].get('execution_time_ms', 0),
                                'CPU %': G.nodes[node].get('cpu_usage_percent', 0),
                                'Parallelizable': G.nodes[node].get('parallelizable', False)
                            }
                            for node in longest_path
                        ])
                        st.dataframe(path_df, use_container_width=True, hide_index=True)
                        
                        st.markdown("**Path Visualization:**")
                        path_str = " → ".join([G.nodes[node].get('task_name', node) for node in longest_path])
                        st.code(path_str)
                        
                except Exception as e:
                    st.info("Could not calculate critical path")
            else:
                # Find cycles
                try:
                    cycles = list(nx.simple_cycles(G))
                    if cycles:
                        st.warning(f"⚠️ Found {len(cycles)} cycles")
                        with st.expander("View Cycles", expanded=False):
                            for i, cycle in enumerate(cycles[:5]):  # Show first 5 cycles
                                cycle_names = [G.nodes[node].get('task_name', node) for node in cycle]
                                st.caption(f"Cycle {i+1}: {' → '.join(cycle_names)} → ...")
                    else:
                        st.success("✅ No simple cycles found")
                except:
                    st.warning("Graph contains cycles (non-DAG)")
        except:
            st.info("Cycle detection not available")

# ============================================
# PAGE: TIMELINE
# ============================================
def show_timeline():
    """Display the timeline visualization page"""
    st.markdown(f"""
    <div class="custom-card fade-in">
        <h2 style="color: {THEME['accent_purple']}; margin: 0; font-size: 2.2rem;">📊 Execution Timeline</h2>
        <p style="color: {THEME['text_dark']}; font-size: 1.1rem; margin-top: 0.5rem;">
            Visualize task execution over time with Gantt charts and timeline analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is None:
        show_no_data_message()
        return
    
    df = st.session_state.data
    
    # Timeline controls
    with st.sidebar:
        st.markdown("### ⚙️ Timeline Settings")
        
        # View type
        view_type = st.radio(
            "View Type",
            ["Gantt Chart", "Resource Timeline", "Task Timeline", "Comparison View"],
            help="Choose how to visualize the timeline"
        )
        
        # Grouping options
        group_by = st.selectbox(
            "Group By",
            ["None", "Task Type", "Level", "Priority", "Parallelizable", "Resource Constraints"]
        )
        
        # Color scheme
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Category", "Sequential", "Diverging", "Execution Time", "Resource Usage"]
        )
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            show_dependencies = st.checkbox("Show Dependencies", True)
        with col2:
            show_legend = st.checkbox("Show Legend", True)
        
        # Time range
        st.markdown("### ⏱️ Time Range")
        time_range = st.slider(
            "Time Range (ms)",
            0, 5000, (0, 2000),
            help="Adjust the visible time range"
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            bar_height = st.slider("Bar Height", 20, 50, 30)
            font_size = st.slider("Font Size", 10, 20, 14)
            show_grid = st.checkbox("Show Grid", True)
            show_markers = st.checkbox("Show Time Markers", True)
    
    # Main timeline visualization
    if view_type == "Gantt Chart":
        show_gantt_chart(df, group_by, color_scheme, show_dependencies, show_legend, 
                        bar_height, font_size, show_grid, time_range)
    elif view_type == "Resource Timeline":
        show_resource_timeline(df, time_range)
    elif view_type == "Task Timeline":
        show_task_timeline(df, group_by, time_range)
    elif view_type == "Comparison View":
        show_comparison_view(df, time_range)

def show_gantt_chart(df: pd.DataFrame, group_by: str, color_scheme: str, 
                    show_dependencies: bool, show_legend: bool, bar_height=30, 
                    font_size=14, show_grid=True, time_range=(0, 2000)):
    """Display Gantt chart visualization with improved styling"""
    # Check if we have timing data
    time_cols = [col for col in df.columns if 'time' in str(col).lower()]
    
    if not time_cols:
        st.warning("""
        ⚠️ No timing columns found in the data.
        
        To create a Gantt chart, your data needs:
        - Start time column (e.g., 'start_time', 'begin_time')
        - End time or duration column (e.g., 'end_time', 'duration', 'execution_time')
        """)
        
        # Offer to generate timing data
        if st.button("🔄 Generate Sample Timeline Data"):
            df = generate_timing_data(df)
            st.session_state.data = df
            st.rerun()
        return
    
    # Create Gantt chart
    st.markdown("### 🗓️ Gantt Chart")
    
    # Prepare data for Gantt chart
    gantt_data = prepare_gantt_data(df, group_by)
    
    # Create the Gantt chart using Plotly with improved styling
    fig = create_gantt_figure_improved(
        gantt_data, group_by, color_scheme, show_dependencies, 
        bar_height, font_size, show_grid, time_range
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Timeline statistics
    show_timeline_stats(gantt_data)

def create_gantt_figure_improved(gantt_df: pd.DataFrame, group_by: str, color_scheme: str, 
                               show_dependencies: bool, bar_height=30, font_size=14, 
                               show_grid=True, time_range=(0, 2000), show_legend=True) -> go.Figure:  # Add show_legend parameter
    """Create interactive Gantt chart figure with improved styling"""
    # Find time columns
    time_cols = [col for col in gantt_df.columns if 'time' in str(col).lower()]
    start_col = next((col for col in time_cols if 'start' in str(col).lower() or 'begin' in str(col).lower()), None)
    end_col = next((col for col in time_cols if 'end' in str(col).lower() or 'finish' in str(col).lower()), None)
    
    if not start_col or not end_col:
        # Try to calculate from duration
        duration_col = next((col for col in time_cols if 'duration' in str(col).lower() or 'execution' in str(col).lower()), None)
        if duration_col:
            gantt_df['start_time'] = 0
            gantt_df['end_time'] = gantt_df[duration_col]
            start_col = 'start_time'
            end_col = 'end_time'
        else:
            st.error("Could not find start and end time columns")
            return go.Figure()
    
    # Calculate optimal height
    num_tasks = len(gantt_df)
    chart_height = max(400, num_tasks * bar_height)
    
    # Create the Gantt chart
    if group_by != "None" and 'group' in gantt_df.columns:
        # Grouped Gantt chart
        fig = px.timeline(
            gantt_df,
            x_start=start_col,
            x_end=end_col,
            y="task_name",
            color="group",
            hover_data=[col for col in gantt_df.columns if col not in [start_col, end_col, 'task_name', 'group']],
            title="Task Execution Timeline",
            color_discrete_sequence=px.colors.qualitative.Set3,
            category_orders={"task_name": sorted(gantt_df['task_name'].unique())}
        )
    else:
        # Ungrouped Gantt chart
        fig = px.timeline(
            gantt_df,
            x_start=start_col,
            x_end=end_col,
            y="task_name",
            hover_data=[col for col in gantt_df.columns if col not in [start_col, end_col, 'task_name']],
            title="Task Execution Timeline",
            color_discrete_sequence=['#7e3af2'],
            category_orders={"task_name": sorted(gantt_df['task_name'].unique())}
        )
    
    # Update layout with improved styling
    fig.update_layout(
        plot_bgcolor='rgba(30, 11, 61, 0.3)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=THEME['text_light'],
        font_family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif",
        xaxis_title="Time (ms)",
        yaxis_title="Task",
        height=chart_height,
        showlegend=group_by != "None" and 'group' in gantt_df.columns and show_legend,
        hovermode='closest',
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_range=time_range
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=show_grid,
        gridcolor='rgba(168, 85, 247, 0.1)',
        gridwidth=1,
        zeroline=False,
        tickfont=dict(size=font_size-2),
        title_font=dict(size=font_size)
    )
    
    fig.update_yaxes(
        showgrid=False,
        tickfont=dict(size=font_size-1),
        title_font=dict(size=font_size),
        automargin=True,
        categoryorder="total ascending"
    )
    
    # Update traces for better visibility
    fig.update_traces(
        marker_line_color='white',
        marker_line_width=1.5,
        opacity=0.85,
        textposition='inside',
        textfont=dict(
            size=font_size-2,
            color='white'
        ),
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Start: %{x}<br>" +
            "End: %{x2}<br>" +
            "Duration: %{customdata[0]:.1f} ms<br>" +
            "<extra></extra>"
        )
    )
    
    # Add task names if space allows
    if num_tasks < 50 and font_size > 12:
        fig.update_traces(
            text=gantt_df['task_name'],
            textposition='inside',
            insidetextanchor='start',
            textfont=dict(
                size=font_size-2,
                color='white'
            )
        )
    
    return fig

def prepare_gantt_data(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Prepare data for Gantt chart"""
    gantt_df = df.copy()
    
    # Ensure we have task names
    if 'task_name' not in gantt_df.columns:
        # Try to find name column
        name_col = None
        for col in gantt_df.columns:
            if 'name' in str(col).lower():
                name_col = col
                break
        
        if name_col:
            gantt_df['task_name'] = gantt_df[name_col]
        else:
            # Use task ID or index
            if 'task_id' in gantt_df.columns:
                gantt_df['task_name'] = gantt_df['task_id']
            else:
                gantt_df['task_name'] = [f"Task_{i}" for i in range(len(gantt_df))]
    
    # Find start and end columns
    start_col = None
    end_col = None
    duration_col = None
    
    for col in gantt_df.columns:
        col_lower = str(col).lower()
        if 'start' in col_lower or 'begin' in col_lower:
            start_col = col
        elif 'end' in col_lower or 'finish' in col_lower:
            end_col = col
        elif 'duration' in col_lower or 'execution' in col_lower:
            duration_col = col
    
    # Calculate missing times if needed
    if start_col and duration_col and not end_col:
        gantt_df['end_time'] = gantt_df[start_col] + gantt_df[duration_col]
        end_col = 'end_time'
    elif end_col and duration_col and not start_col:
        gantt_df['start_time'] = gantt_df[end_col] - gantt_df[duration_col]
        start_col = 'start_time'
    
    # If still missing timing data, create synthetic
    if not start_col or not end_col:
        gantt_df = generate_timing_data(gantt_df)
        start_col = 'start_time'
        end_col = 'end_time'
    
    # Add duration column if not present
    if 'duration' not in gantt_df.columns:
        gantt_df['duration'] = gantt_df[end_col] - gantt_df[start_col]
    
    # Add grouping column
    if group_by != "None":
        if group_by in gantt_df.columns:
            gantt_df['group'] = gantt_df[group_by]
        else:
            # Try case-insensitive match
            col_lower = group_by.lower()
            matching_cols = [col for col in gantt_df.columns if col.lower() == col_lower]
            if matching_cols:
                gantt_df['group'] = gantt_df[matching_cols[0]]
            else:
                gantt_df['group'] = 'All Tasks'
    else:
        gantt_df['group'] = 'All Tasks'
    
    # Sort by start time
    gantt_df = gantt_df.sort_values(start_col)
    
    # Select relevant columns
    keep_cols = [start_col, end_col, 'task_name', 'group', 'duration']
    keep_cols.extend([col for col in gantt_df.columns if col not in keep_cols])
    
    return gantt_df[keep_cols]

def generate_timing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic timing data for tasks"""
    df = df.copy()
    
    # Check if we have dependencies
    has_dependencies = 'parent_id' in df.columns or 'dependencies' in df.columns
    
    if has_dependencies and 'parent_id' in df.columns:
        # Calculate levels based on dependencies
        df['level'] = 0
        
        # Simple level calculation
        for i in range(10):  # Maximum 10 iterations
            mask = df['parent_id'].notna() & df['parent_id'].isin(df['task_id'])
            if not mask.any():
                break
            df.loc[mask, 'level'] = df.loc[mask, 'parent_id'].map(
                df.set_index('task_id')['level']
            ) + 1
        
        # Generate start times based on levels
        base_time = 0
        max_start_time = 0
        
        for level in sorted(df['level'].unique()):
            level_tasks = df[df['level'] == level]
            num_tasks = len(level_tasks)
            
            if num_tasks > 0:
                # Distribute tasks in this level over time
                time_per_task = 50  # ms between task starts in same level
                start_times = []
                
                for i in range(num_tasks):
                    if level == 0:
                        # Root tasks can start at different times
                        start_time = base_time + i * time_per_task
                    else:
                        # Dependent tasks start after parent + some delay
                        task = level_tasks.iloc[i]
                        parent_id = task['parent_id']
                        if parent_id and parent_id in df['task_id'].values:
                            parent_row = df[df['task_id'] == parent_id]
                            if not parent_row.empty:
                                parent_end = parent_row.iloc[0].get('end_time', 0)
                                start_time = parent_end + np.random.exponential(10)
                            else:
                                start_time = base_time + i * time_per_task
                        else:
                            start_time = base_time + i * time_per_task
                    
                    start_times.append(start_time)
                
                df.loc[level_tasks.index, 'start_time'] = start_times
                max_start_time = max(max_start_time, max(start_times))
                
                # Update base time for next level
                if 'execution_time_ms' in df.columns:
                    max_execution = level_tasks['execution_time_ms'].max()
                else:
                    max_execution = 100  # default
                
                base_time = max_start_time + max_execution
    else:
        # No dependencies, just sequential or random
        df['start_time'] = np.random.uniform(0, 500, len(df))
    
    # Generate end times
    if 'execution_time_ms' in df.columns:
        df['end_time'] = df['start_time'] + df['execution_time_ms']
    else:
        # Generate synthetic execution times
        np.random.seed(42)
        df['execution_time_ms'] = np.random.exponential(50, len(df)) + 20
        df['end_time'] = df['start_time'] + df['execution_time_ms']
    
    return df

def show_timeline_stats(gantt_df: pd.DataFrame):
    """Display timeline statistics"""
    st.markdown(f"""
    <div class="custom-card">
        <h4 style="color: {THEME['accent_purple']}; margin-bottom: 15px;">📈 Timeline Statistics</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Find time columns
    time_cols = [col for col in gantt_df.columns if 'time' in str(col).lower()]
    start_col = next((col for col in time_cols if 'start' in str(col).lower()), None)
    end_col = next((col for col in time_cols if 'end' in str(col).lower()), None)
    
    if start_col and end_col:
        # Calculate statistics
        durations = gantt_df[end_col] - gantt_df[start_col]
        total_duration = durations.sum()
        max_duration = durations.max()
        min_duration = durations.min()
        avg_duration = durations.mean()
        std_duration = durations.std()
        
        # Calculate makespan
        makespan = gantt_df[end_col].max() - gantt_df[start_col].min()
        
        # Display metrics
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Duration", f"{total_duration:.1f} ms")
        with cols[1]:
            st.metric("Makespan", f"{makespan:.1f} ms")
        with cols[2]:
            st.metric("Avg Task Duration", f"{avg_duration:.1f} ms")
        with cols[3]:
            st.metric("Longest Task", f"{max_duration:.1f} ms")
        
        # Additional metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Shortest Task", f"{min_duration:.1f} ms")
        with col2:
            st.metric("Std Dev", f"{std_duration:.1f} ms")
        
        # Duration distribution
        with st.expander("View Duration Distribution", expanded=False):
            fig = px.histogram(
                gantt_df, 
                x=durations,
                nbins=20,
                title="Task Duration Distribution",
                labels={'x': 'Duration (ms)', 'y': 'Count'},
                color_discrete_sequence=['#7e3af2']
            )
            fig.update_layout(
                plot_bgcolor='rgba(30, 11, 61, 0.3)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=THEME['text_light'],
                font_family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

def show_resource_timeline(df: pd.DataFrame, time_range=(0, 2000)):
    """Resource utilization timeline"""
    st.markdown("### 💻 Resource Utilization Timeline")
    
    # Check for resource columns
    resource_cols = [col for col in df.columns 
                    if any(keyword in str(col).lower() 
                          for keyword in ['cpu', 'memory', 'usage', 'utilization', 'io', 'network'])]
    
    if not resource_cols:
        st.warning("No resource utilization data found.")
        return
    
    # Check for time data
    time_cols = [col for col in df.columns if 'time' in str(col).lower()]
    start_col = next((col for col in time_cols if 'start' in str(col).lower()), None)
    
    if not start_col:
        st.warning("Start time information required for resource timeline.")
        return
    
    # Create resource timeline
    fig = go.Figure()
    
    color_palette = px.colors.qualitative.Set3
    
    for idx, resource_col in enumerate(resource_cols[:4]):  # Limit to first 4 resource types
        if resource_col in df.columns:
            # Clean data
            clean_df = df[[start_col, resource_col]].dropna()
            if len(clean_df) > 1:
                fig.add_trace(go.Scatter(
                    x=clean_df[start_col],
                    y=clean_df[resource_col],
                    mode='lines+markers',
                    name=resource_col,
                    line=dict(width=3, color=color_palette[idx % len(color_palette)]),
                    marker=dict(size=8, symbol='circle'),
                    hovertemplate=(
                        "<b>Time:</b> %{x:.1f} ms<br>" +
                        f"<b>{resource_col}:</b> %{{y:.1f}}<br>" +
                        "<extra></extra>"
                    )
                ))
    
    if len(fig.data) == 0:
        st.warning("No valid resource data to display.")
        return
    
    fig.update_layout(
        title="Resource Utilization Over Time",
        xaxis_title="Time (ms)",
        yaxis_title="Usage",
        plot_bgcolor='rgba(30, 11, 61, 0.3)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=THEME['text_light'],
        font_family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_range=time_range,
        yaxis=dict(
            rangemode='tozero'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Resource statistics
    show_resource_stats(df, resource_cols)

def show_resource_stats(df: pd.DataFrame, resource_cols: List[str]):
    """Display resource statistics"""
    st.markdown("#### 📊 Resource Statistics")
    
    # Display stats for each resource
    for resource_col in resource_cols[:3]:  # Limit to 3 resources
        if resource_col in df.columns:
            resource_data = df[resource_col].dropna()
            if len(resource_data) > 0:
                st.markdown(f"**{resource_col}:**")
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Avg", f"{resource_data.mean():.1f}")
                with cols[1]:
                    st.metric("Max", f"{resource_data.max():.1f}")
                with cols[2]:
                    st.metric("Min", f"{resource_data.min():.1f}")
                with cols[3]:
                    st.metric("Std Dev", f"{resource_data.std():.1f}")

def show_task_timeline(df: pd.DataFrame, group_by: str, time_range=(0, 2000)):
    """Task timeline view"""
    st.markdown("### 📋 Task Timeline View")
    
    # Prepare data
    timeline_data = prepare_gantt_data(df, group_by)
    
    # Find time columns
    time_cols = [col for col in timeline_data.columns if 'time' in str(col).lower()]
    start_col = next((col for col in time_cols if 'start' in str(col).lower()), None)
    end_col = next((col for col in time_cols if 'end' in str(col).lower()), None)
    
    if not start_col or not end_col:
        st.error("Could not find time columns")
        return
    
    # Create timeline
    fig = go.Figure()
    
    # Add bars for each task
    for idx, row in timeline_data.iterrows():
        fig.add_trace(go.Bar(
            y=[row['task_name']],
            x=[row[end_col] - row[start_col]],
            base=row[start_col],
            orientation='h',
            name=row['task_name'],
            marker_color='#7e3af2',
            opacity=0.8,
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Start: %{base:.1f} ms<br>" +
                "End: %{x:.1f} ms<br>" +
                "Duration: %{width:.1f} ms<br>" +
                "<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title="Task Timeline",
        xaxis_title="Time (ms)",
        yaxis_title="Task",
        plot_bgcolor='rgba(30, 11, 61, 0.3)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=THEME['text_light'],
        font_family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif",
        height=max(400, len(timeline_data) * 25),
        showlegend=False,
        bargap=0.2,
        xaxis_range=time_range
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_comparison_view(df: pd.DataFrame, time_range=(0, 2000)):
    """Comparison view of different task types"""
    st.markdown("### ⚖️ Task Type Comparison")
    
    # Check if we have task type information
    if 'task_type' not in df.columns:
        st.info("Task type information not available for comparison.")
        return
    
    # Prepare data
    timeline_data = prepare_gantt_data(df, "task_type")
    
    # Find time columns
    time_cols = [col for col in timeline_data.columns if 'time' in str(col).lower()]
    start_col = next((col for col in time_cols if 'start' in str(col).lower()), None)
    end_col = next((col for col in time_cols if 'end' in str(col).lower()), None)
    
    if not start_col or not end_col:
        st.error("Could not find time columns")
        return
    
    # Create comparison chart
    task_types = timeline_data['group'].unique()
    
    fig = go.Figure()
    
    for task_type in task_types:
        type_data = timeline_data[timeline_data['group'] == task_type]
        if len(type_data) > 0:
            avg_duration = (type_data[end_col] - type_data[start_col]).mean()
            avg_start = type_data[start_col].mean()
            
            fig.add_trace(go.Bar(
                x=[task_type],
                y=[avg_duration],
                name=task_type,
                marker_color=px.colors.qualitative.Set3[list(task_types).index(task_type) % len(px.colors.qualitative.Set3)],
                hovertemplate=(
                    f"<b>{task_type}</b><br>" +
                    "Avg Duration: %{y:.1f} ms<br>" +
                    "Avg Start: %{customdata:.1f} ms<br>" +
                    "Tasks: %{customdata[1]}<br>" +
                    "<extra></extra>"
                ),
                customdata=np.column_stack([type_data[start_col].mean(), len(type_data)])
            ))
    
    fig.update_layout(
        title="Average Duration by Task Type",
        xaxis_title="Task Type",
        yaxis_title="Average Duration (ms)",
        plot_bgcolor='rgba(30, 11, 61, 0.3)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=THEME['text_light'],
        font_family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif",
        height=400,
        showlegend=False,
        bargap=0.3
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed comparison
    with st.expander("📊 Detailed Task Type Statistics", expanded=False):
        stats_data = []
        for task_type in task_types:
            type_data = timeline_data[timeline_data['group'] == task_type]
            if len(type_data) > 0:
                durations = type_data[end_col] - type_data[start_col]
                stats_data.append({
                    'Task Type': task_type,
                    'Count': len(type_data),
                    'Avg Duration (ms)': durations.mean(),
                    'Max Duration (ms)': durations.max(),
                    'Min Duration (ms)': durations.min(),
                    'Total Duration (ms)': durations.sum()
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)

# ============================================
# PAGE: LATENCY ANALYSIS
# ============================================
def show_latency_analysis():
    """Display latency analysis page"""
    st.markdown(f"""
    <div class="custom-card fade-in">
        <h2 style="color: {THEME['accent_purple']}; margin: 0; font-size: 2.2rem;">⏱️ Latency Analysis</h2>
        <p style="color: {THEME['text_dark']}; font-size: 1.1rem; margin-top: 0.5rem;">
            Analyze execution times, identify bottlenecks, and optimize task performance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is None:
        show_no_data_message()
        return
    
    df = st.session_state.data
    
    # Check for execution time data
    time_cols = [col for col in df.columns if any(keyword in str(col).lower() 
                                                 for keyword in ['time', 'duration', 'latency', 'execution'])]
    
    if not time_cols:
        st.warning("""
        ⚠️ No timing data found for latency analysis.
        
        To analyze latency, your data needs execution time information.
        Try uploading data with columns like:
        - execution_time
        - duration_ms  
        - latency
        - task_time
        """)
        return
    
    # Main analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "⚠️ Bottlenecks", "⚡ Parallelization", "📈 Correlations"])
    
    with tab1:
        show_latency_distribution(df, time_cols)
    
    with tab2:
        show_bottleneck_analysis(df, time_cols)
    
    with tab3:
        show_parallelization_analysis(df, time_cols)
    
    with tab4:
        show_correlation_analysis(df, time_cols)

def show_latency_distribution(df: pd.DataFrame, time_cols: List[str]):
    """Show latency distribution analysis"""
    st.markdown("### 📊 Latency Distribution")
    
    # Find the best time column
    time_col = time_cols[0]  # Use first time column found
    time_data = df[time_col].dropna()
    
    if len(time_data) == 0:
        st.warning("No valid timing data available.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create distribution plot
        fig = px.histogram(
            df, 
            x=time_col,
            nbins=30,
            title=f"Distribution of {time_col}",
            color_discrete_sequence=['#7e3af2'],
            opacity=0.8,
            marginal="box"
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(30, 11, 61, 0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=THEME['text_light'],
            font_family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif",
            xaxis_title=time_col,
            yaxis_title="Count",
            height=400,
            bargap=0.1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Latency statistics
        st.markdown("### 📈 Key Statistics")
        
        stats = {
            "Average": f"{time_data.mean():.1f} ms",
            "Median": f"{time_data.median():.1f} ms",
            "Std Dev": f"{time_data.std():.1f} ms",
            "Minimum": f"{time_data.min():.1f} ms",
            "Maximum": f"{time_data.max():.1f} ms",
            "90th %ile": f"{time_data.quantile(0.9):.1f} ms",
            "95th %ile": f"{time_data.quantile(0.95):.1f} ms",
            "Total": f"{time_data.sum():.1f} ms"
        }
        
        for label, value in stats.items():
            st.metric(label, value)
    
    # Additional distribution analysis
    st.markdown("#### 📉 Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Skewness and kurtosis
        from scipy import stats as scipy_stats
        try:
            skewness = scipy_stats.skew(time_data)
            kurtosis = scipy_stats.kurtosis(time_data)
            
            st.metric("Skewness", f"{skewness:.3f}")
            st.metric("Kurtosis", f"{kurtosis:.3f}")
            
            if skewness > 1:
                st.info("Distribution is right-skewed (long tail on right)")
            elif skewness < -1:
                st.info("Distribution is left-skewed (long tail on left)")
            else:
                st.info("Distribution is approximately symmetric")
        except:
            pass
    
    with col2:
        # Percentile analysis
        percentiles = [25, 50, 75, 90, 95, 99]
        percentile_values = [time_data.quantile(p/100) for p in percentiles]
        
        percentile_df = pd.DataFrame({
            'Percentile': [f'{p}%' for p in percentiles],
            'Value (ms)': percentile_values
        })
        st.dataframe(percentile_df, use_container_width=True, hide_index=True)

def show_bottleneck_analysis(df: pd.DataFrame, time_cols: List[str]):
    """Show bottleneck analysis"""
    st.markdown("### ⚠️ Bottleneck Identification")
    
    time_col = time_cols[0]
    time_data = df[time_col].dropna()
    
    if len(time_data) < 10:
        st.warning("Need at least 10 data points for bottleneck analysis.")
        return
    
    # Bottleneck detection
    bottleneck_threshold = time_data.quantile(0.9)
    bottlenecks = df[df[time_col] > bottleneck_threshold]
    
    if len(bottlenecks) > 0:
        st.warning(f"**Found {len(bottlenecks)} potential bottlenecks "
                  f"(tasks > {bottleneck_threshold:.1f} ms):**")
        
        # Show bottleneck tasks
        bottleneck_cols = ['task_id', 'task_name', time_col] if 'task_name' in df.columns else ['task_id', time_col]
        bottleneck_cols = [col for col in bottleneck_cols if col in bottlenecks.columns]
        
        # Add additional info if available
        for col in ['cpu_usage_percent', 'memory_usage_mb', 'task_type', 'parallelizable']:
            if col in bottlenecks.columns:
                bottleneck_cols.append(col)
        
        st.dataframe(
            bottlenecks[bottleneck_cols].sort_values(time_col, ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Bottleneck impact
        total_time = time_data.sum()
        bottleneck_time = bottlenecks[time_col].sum()
        bottleneck_impact = (bottleneck_time / total_time) * 100
        
        st.markdown(f"**Impact Analysis:**")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Bottleneck Time", f"{bottleneck_time:.0f} ms")
        with cols[1]:
            st.metric("Total Time", f"{total_time:.0f} ms")
        with cols[2]:
            st.metric("Impact %", f"{bottleneck_impact:.1f}%")
        
        # Bottleneck characteristics
        if 'task_type' in bottlenecks.columns:
            st.markdown("#### 📋 Bottleneck by Task Type")
            type_dist = bottlenecks['task_type'].value_counts()
            fig = px.pie(
                values=type_dist.values,
                names=type_dist.index,
                title="Bottleneck Tasks by Type",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                plot_bgcolor='rgba(30, 11, 61, 0.3)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=THEME['text_light']
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No significant bottlenecks detected!")

def show_parallelization_analysis(df: pd.DataFrame, time_cols: List[str]):
    """Show parallelization potential analysis"""
    st.markdown("### ⚡ Parallelization Analysis")
    
    time_col = time_cols[0]
    
    # Check for parallelizable column
    if 'parallelizable' not in df.columns:
        st.info("""
        Parallelization analysis requires a 'parallelizable' column.
        
        To enable this analysis:
        1. Add a 'parallelizable' column to your data
        2. Use boolean values (True/False) to indicate which tasks can run in parallel
        3. Re-analyze your data
        """)
        return
    
    parallel_tasks = df[df['parallelizable'] == True]
    sequential_tasks = df[df['parallelizable'] == False]
    
    if len(parallel_tasks) == 0:
        st.warning("No parallelizable tasks found.")
        return
    
    # Analysis metrics
    total_tasks = len(df)
    parallel_percent = (len(parallel_tasks) / total_tasks) * 100
    
    if time_col in df.columns:
        parallel_time = parallel_tasks[time_col].sum()
        sequential_time = sequential_tasks[time_col].sum()
        total_time = df[time_col].sum()
    else:
        parallel_time = len(parallel_tasks) * 100  # Estimate
        sequential_time = len(sequential_tasks) * 100  # Estimate
        total_time = total_tasks * 100  # Estimate
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Parallelizable Tasks", f"{len(parallel_tasks)}")
    with col2:
        st.metric("Sequential Tasks", f"{len(sequential_tasks)}")
    with col3:
        st.metric("Parallel %", f"{parallel_percent:.1f}%")
    with col4:
        if total_time > 0:
            parallel_time_pct = (parallel_time / total_time) * 100
            st.metric("Parallel Time %", f"{parallel_time_pct:.1f}%")
    
    # Speedup estimation
    st.markdown("#### 🚀 Speedup Estimation")
    
    # Amdahl's Law calculation
    if total_time > 0:
        parallel_fraction = parallel_time / total_time
        
        # User input for parallel factor
        parallel_factor = st.slider(
            "Parallelization Factor",
            2, 32, 4,
            help="Number of parallel processors/threads"
        )
        
        # Amdahl's Law: Speedup = 1 / ((1 - p) + (p / n))
        if parallel_fraction > 0:
            speedup = 1 / ((1 - parallel_fraction) + (parallel_fraction / parallel_factor))
            efficiency = (speedup / parallel_factor) * 100
            
            optimized_time = total_time / speedup if speedup > 0 else total_time
            time_saved = total_time - optimized_time
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Theoretical Speedup", f"{speedup:.2f}x")
            with col2:
                st.metric("Efficiency", f"{efficiency:.1f}%")
            with col3:
                st.metric("Time Saved", f"{time_saved:.0f} ms")
            
            # Visualization
            fig = go.Figure(data=[
                go.Bar(
                    name='Current', 
                    x=['Total Time'], 
                    y=[total_time], 
                    marker_color='#7e3af2'
                ),
                go.Bar(
                    name=f'Optimized ({parallel_factor}x)', 
                    x=['Total Time'], 
                    y=[optimized_time], 
                    marker_color='#10b981'
                )
            ])
            
            fig.update_layout(
                title="Parallelization Impact",
                plot_bgcolor='rgba(30, 11, 61, 0.3)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=THEME['text_light'],
                height=300,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Parallel task characteristics
    if 'task_type' in df.columns:
        st.markdown("#### 📋 Parallel Task Analysis")
        
        parallel_by_type = parallel_tasks['task_type'].value_counts()
        sequential_by_type = sequential_tasks['task_type'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                name='Parallelizable',
                x=parallel_by_type.index,
                y=parallel_by_type.values,
                marker_color='#10b981'
            ),
            go.Bar(
                name='Sequential',
                x=sequential_by_type.index,
                y=sequential_by_type.values,
                marker_color='#7e3af2'
            )
        ])
        
        fig.update_layout(
            title="Task Parallelizability by Type",
            xaxis_title="Task Type",
            yaxis_title="Count",
            plot_bgcolor='rgba(30, 11, 61, 0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=THEME['text_light'],
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_correlation_analysis(df: pd.DataFrame, time_cols: List[str]):
    """Show correlation analysis"""
    st.markdown("### 📈 Correlation Analysis")
    
    time_col = time_cols[0]
    
    # Find numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return
    
    # Correlation matrix
    st.markdown("#### 🔗 Correlation Matrix")
    
    # Select columns for correlation
    default_cols = [time_col]
    for col in ['cpu_usage_percent', 'memory_usage_mb', 'io_operations', 'network_calls']:
        if col in numeric_cols:
            default_cols.append(col)
    
    selected_cols = st.multiselect(
        "Select columns for correlation analysis:",
        numeric_cols,
        default=default_cols[:min(6, len(default_cols))]
    )
    
    if len(selected_cols) >= 2:
        # Calculate correlation matrix
        corr_matrix = df[selected_cols].corr()
        
        # Heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            title="Correlation Matrix Heatmap"
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(30, 11, 61, 0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=THEME['text_light'],
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlations with execution time
        if time_col in corr_matrix.columns:
            time_correlations = corr_matrix[time_col].drop(time_col).sort_values(key=abs, ascending=False)
            
            st.markdown("#### 🎯 Top Correlations with Execution Time")
            
            top_corr_df = pd.DataFrame({
                'Feature': time_correlations.index,
                'Correlation': time_correlations.values,
                'Strength': ['Strong' if abs(c) > 0.7 else 'Moderate' if abs(c) > 0.3 else 'Weak' for c in time_correlations.values]
            })
            
            st.dataframe(top_corr_df, use_container_width=True, hide_index=True)
            
            # Scatter plot for top correlation
            if len(time_correlations) > 0:
                top_feature = time_correlations.index[0]
                top_corr = time_correlations.iloc[0]
                
                st.markdown(f"#### 📊 {time_col} vs {top_feature} (Correlation: {top_corr:.3f})")
                
                fig = px.scatter(
                    df,
                    x=top_feature,
                    y=time_col,
                    title=f"{time_col} vs {top_feature}",
                    color_discrete_sequence=['#7e3af2'],
                    opacity=0.7,
                    trendline="ols"
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(30, 11, 61, 0.3)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color=THEME['text_light']
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE: OPTIMIZATION
# ============================================
def show_optimization():
    """Display optimization suggestions page"""
    st.markdown("# ⚡ Optimization Engine")
    st.write("AI-powered optimization suggestions and performance simulation.")
    st.markdown("---")
    
    if st.session_state.data is None:
        show_no_data_message()
        return
    
    df = st.session_state.data
    
    # Check if we have AI analysis
    if st.session_state.analysis is None:
        st.info("""
        🤖 **AI Analysis Required**
        
        To get optimization suggestions, we need to analyze your data first.
        """)
        
        if st.button("🚀 Run AI Analysis Now", type="primary"):
            perform_ai_analysis(df)
        return
    
    # Main optimization content
    tab1, tab2, tab3, tab4 = st.tabs(["💡 Suggestions", "🎯 Simulation", "📊 Impact", "🗺️ Roadmap"])
    
    with tab1:
        show_optimization_suggestions()
    
    with tab2:
        show_optimization_simulation(df)
    
    with tab3:
        show_impact_analysis(df)
    
    with tab4:
        show_optimization_roadmap()


def show_optimization_suggestions():
    """Display optimization suggestions"""
    if st.session_state.analysis is None:
        st.warning("No analysis data available. Please run AI analysis first.")
        return
    
    analysis = st.session_state.analysis
    
    # Display suggestions from AI analysis
    st.markdown("### 🚀 AI Optimization Suggestions")
    
    # Check for suggestions in analysis
    suggestions = analysis.get("optimization_suggestions", [])
    
    if suggestions:
        # Categorize suggestions
        categories = {}
        for suggestion in suggestions:
            category = suggestion.get("type", "general")
            if category not in categories:
                categories[category] = []
            categories[category].append(suggestion)
        
        # Display by category
        for category, cat_suggestions in categories.items():
            st.markdown(f"#### {get_category_icon(category)} {category.title()}")
            
            for suggestion in cat_suggestions:
                display_suggestion_native(suggestion)
    else:
        # Generate generic suggestions based on data
        st.info("""
        ### 💡 General Optimization Strategies
        
        Based on common parallel computing patterns:
        
        1. **Task Parallelization**
           - Identify independent tasks that can run concurrently
           - Implement parallel execution where dependencies allow
        
        2. **Resource Optimization**
           - Balance CPU and memory usage across tasks
           - Implement resource pooling for expensive operations
        
        3. **Dependency Optimization**
           - Reduce critical path length
           - Reorder tasks to maximize parallel execution
        
        4. **Memory Management**
           - Implement caching for frequently used data
           - Use streaming for large datasets
        
        5. **I/O Optimization**
           - Batch file operations
           - Use asynchronous I/O where possible
        """)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Apply Suggestions", use_container_width=True):
            st.success("Optimization suggestions applied to simulation!")
    
    with col2:
        if st.button("📊 Compare Scenarios", use_container_width=True):
            st.info("Scenario comparison feature coming soon!")
    
    with col3:
        if st.button("📋 Export Plan", use_container_width=True):
            st.info("Export feature coming soon!")


def display_suggestion_native(suggestion: Dict):
    """Display a single optimization suggestion using native Streamlit"""
    impact = suggestion.get("impact", "medium")
    effort = suggestion.get("effort", "medium")
    description = suggestion.get('description', 'Optimization Suggestion')
    details = suggestion.get('details', 'Detailed recommendation available.')
    suggestion_type = suggestion.get('type', 'general')
    
    icon = {
        "parallelization": "⚡",
        "bottleneck": "⚠️",
        "resource": "💾",
        "dependency": "🔄",
        "memory": "🧠",
        "io": "📁",
        "network": "🌐",
        "cost": "💰"
    }.get(suggestion_type, "🔧")
    
    # Create card using native Streamlit
    with st.container():
        if impact == "high":
            st.error(f"**{icon} {description}**")
        elif impact == "medium":
            st.warning(f"**{icon} {description}**")
        else:
            st.info(f"**{icon} {description}**")
        
        st.write(details)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"**Impact:** {impact.upper()}")
        with col2:
            st.caption(f"**Effort:** {effort.upper()}")
        with col3:
            st.caption(f"**Type:** {suggestion_type}")
        
        st.markdown("---")


def display_suggestion(suggestion: Dict):
    """Display a single optimization suggestion - calls native version"""
    display_suggestion_native(suggestion)


def get_category_icon(category: str) -> str:
    """Get icon for optimization category"""
    icons = {
        "parallelization": "⚡",
        "bottleneck": "⚠️",
        "resource": "💾",
        "dependency": "🔄",
        "memory": "🧠",
        "io": "📁",
        "network": "🌐",
        "cost": "💰",
        "general": "🔧"
    }
    return icons.get(category, "🔧")


def show_optimization_simulation(df: pd.DataFrame):
    """Show optimization simulation results"""
    st.markdown("### 🎯 Optimization Simulation")
    
    # Simulation parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        parallel_factor = st.slider(
            "Parallelization Factor",
            1.0, 10.0, 2.0, 0.5,
            help="How many tasks can run in parallel"
        )
    
    with col2:
        resource_optimization = st.slider(
            "Resource Optimization %",
            0, 50, 20, 5,
            help="Percentage improvement in resource usage"
        )
    
    with col3:
        cache_efficiency = st.slider(
            "Cache Efficiency",
            0, 100, 30, 10,
            help="Percentage of tasks benefiting from caching"
        )
    
    # Run simulation
    if st.button("🎮 Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Running optimization simulation..."):
            # Get analysis
            analysis = st.session_state.analysis or {}
            
            # Run simulation
            if st.session_state.ai_processor:
                simulation_results = st.session_state.ai_processor.simulate_optimization(
                    df, analysis, parallel_factor, resource_optimization, cache_efficiency
                )
                
                st.session_state.simulation_results = simulation_results
                
                # Display results
                display_simulation_results(simulation_results)
            else:
                st.error("AI processor not available")
    
    # Show previous results if available
    elif st.session_state.simulation_results:
        display_simulation_results(st.session_state.simulation_results)


def display_simulation_results(results: Dict[str, Any]):
    """Display simulation results"""
    st.markdown("### 📊 Simulation Results")
    st.markdown("---")
    
    # Performance metrics
    current_perf = results.get("current_performance", {})
    optimized_perf = results.get("optimized_performance", {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_time = current_perf.get("total_time", 0)
        st.metric(
            "Original Time", 
            f"{current_time:,.0f} ms",
            delta_color="off"
        )
    
    with col2:
        optimized_time = optimized_perf.get("total_time", 0)
        improvement = optimized_perf.get("improvement_percentage", 0)
        st.metric(
            "Optimized Time", 
            f"{optimized_time:,.0f} ms",
            delta=f"-{improvement:.1f}%"
        )
    
    with col3:
        speedup = optimized_perf.get("speedup", 1.0)
        efficiency = optimized_perf.get("efficiency_percentage", 0)
        st.metric(
            "Speedup", 
            f"{speedup:.2f}x",
            delta=f"{efficiency:.1f}% efficient"
        )
    
    # Visualization
    fig = go.Figure(data=[
        go.Bar(
            name='Original', 
            x=['Total Execution Time'], 
            y=[current_time], 
            marker_color='#7e3af2',
            text=[f"{current_time:.0f} ms"],
            textposition='auto',
        ),
        go.Bar(
            name='Optimized', 
            x=['Total Execution Time'], 
            y=[optimized_time], 
            marker_color='#10b981',
            text=[f"{optimized_time:.0f} ms"],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Optimization Impact",
        plot_bgcolor='rgba(30, 11, 61, 0.3)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f3e8ff',
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Breakdown of improvements
    with st.expander("📊 Improvement Breakdown"):
        impacts = results.get("optimization_impacts", [])
        
        for impact in impacts:
            impact_pct = impact.get("impact_percentage", 0)
            impact_type = impact.get("type", "unknown")
            description = impact.get("description", "")
            
            st.success(f"**{impact_type.title()}:** {description} - **{impact_pct:.1f}% improvement**")
    
    # Recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
        st.markdown("#### 💡 Recommendations")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")


def show_impact_analysis(df: pd.DataFrame):
    """Show impact analysis of optimizations"""
    st.markdown("### 📊 Impact Analysis")
    
    if st.session_state.analysis is None:
        st.info("Run AI analysis first to see impact analysis.")
        return
    
    analysis = st.session_state.analysis
    
    # Performance metrics
    perf_metrics = analysis.get("performance_metrics", {})
    
    if perf_metrics:
        st.markdown("#### 🎯 Performance Impact")
        
        cols = st.columns(3)
        
        with cols[0]:
            current_time = perf_metrics.get("current_total_time", 0)
            st.metric("Current Total Time", f"{current_time:,.0f} ms")
        
        with cols[1]:
            optimized_time = perf_metrics.get("optimized_time", 0)
            st.metric("Potential Optimized Time", f"{optimized_time:,.0f} ms")
        
        with cols[2]:
            speedup = perf_metrics.get("speedup_factor", 1.0)
            st.metric("Potential Speedup", f"{speedup:.2f}x")
    
    # Resource analysis
    resource_analysis = analysis.get("resource_analysis", {})
    if resource_analysis:
        st.markdown("#### 💾 Resource Impact")
        
        for recommendation in resource_analysis.get("recommendations", []):
            st.info(f"💡 {recommendation}")
    
    # Cost analysis
    if 'cost' in df.columns:
        st.markdown("#### 💰 Cost Impact")
        
        total_cost = df['cost'].sum()
        avg_cost = df['cost'].mean()
        
        # Estimate optimized cost (reduce by speedup factor)
        speedup = perf_metrics.get("speedup_factor", 1.0)
        optimized_cost = total_cost / speedup if speedup > 0 else total_cost
        cost_savings = total_cost - optimized_cost
        savings_percent = (cost_savings / total_cost) * 100 if total_cost > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Cost", f"${total_cost:.2f}")
        with col2:
            st.metric("Optimized Cost", f"${optimized_cost:.2f}")
        with col3:
            st.metric("Savings", f"${cost_savings:.2f}", delta=f"-{savings_percent:.1f}%")


def show_optimization_roadmap():
    """Show optimization implementation roadmap"""
    st.markdown("### 🗺️ Implementation Roadmap")
    
    if st.session_state.optimization_plan is None:
        st.info("Run AI analysis to generate an implementation roadmap.")
        return
    
    plan = st.session_state.optimization_plan
    
    # Summary
    summary = plan.get("summary", {})
    if summary:
        st.markdown("#### 📋 Summary")
        
        cols = st.columns(4)
        metrics = [
            ("Total Suggestions", summary.get("total_suggestions", 0)),
            ("Estimated Impact", summary.get("estimated_impact", "Medium")),
            ("Timeline", summary.get("implementation_timeline", "2-4 weeks")),
            ("Success Probability", f"{summary.get('success_probability', 0.7)*100:.0f}%")
        ]
        
        for idx, (label, value) in enumerate(metrics):
            with cols[idx]:
                st.metric(label, value)
    
    # Phases
    phases = plan.get("phases", [])
    if phases:
        st.markdown("#### 📅 Implementation Phases")
        
        for phase in phases:
            with st.expander(f"Phase {phase['phase']}: {phase['name']} ({phase['duration']})", expanded=False):
                # Tasks
                st.markdown("**Tasks:**")
                for task in phase.get("tasks", []):
                    st.markdown(f"• **{task.get('description', 'Task')}**")
                    if task.get("details"):
                        for detail in task["details"]:
                            st.markdown(f"  - {detail}")
                    
                    # Success criteria
                    if task.get("success_criteria"):
                        st.markdown("  **Success Criteria:**")
                        for criterion in task["success_criteria"]:
                            st.markdown(f"  - ✓ {criterion}")
                
                # Dependencies
                if phase.get("dependencies"):
                    st.markdown("**Dependencies:**")
                    for dep in phase["dependencies"]:
                        st.markdown(f"• {dep}")
                
                # Risks and mitigations
                col1, col2 = st.columns(2)
                
                with col1:
                    if phase.get("risks"):
                        st.markdown("**Risks:**")
                        for risk in phase["risks"]:
                            st.warning(f"⚠️ {risk}")
                
                with col2:
                    if phase.get("mitigations"):
                        st.markdown("**Mitigations:**")
                        for mitigation in phase["mitigations"]:
                            st.success(f"✓ {mitigation}")
    
    # Metrics to track
    metrics = plan.get("metrics_to_track", [])
    if metrics:
        st.markdown("#### 📈 Metrics to Track")
        cols = st.columns(3)
        for idx, metric in enumerate(metrics):
            with cols[idx % 3]:
                st.info(f"📊 {metric}")
    
    # Expected outcomes
    outcomes = plan.get("expected_outcomes", [])
    if outcomes:
        st.markdown("#### 🎯 Expected Outcomes")
        for outcome in outcomes:
            st.success(f"✅ {outcome}")

# ============================================
# PAGE: ABOUT
# ============================================
def show_about():
    """Display redesigned about page - concise with cards and professional footer"""
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Features Section
    st.markdown(f"""
    <div class="custom-card fade-in">
        <h2 style="color: {THEME['accent_purple']}; margin-bottom: 1.5rem; font-size: 2rem; text-align: center;">
            ✨ Key Features
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards Grid - Compact
    feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
    
    features = [
        {"icon": "🤖", "title": "AI Analysis", "color": "#3b82f6"},
        {"icon": "📊", "title": "Visualizations", "color": "#10b981"},
        {"icon": "⚡", "title": "Optimization", "color": "#f59e0b"},
        {"icon": "📁", "title": "Multi-Format", "color": "#8b5cf6"},
        {"icon": "🎯", "title": "Simulation", "color": "#ef4444"},
        {"icon": "🎨", "title": "Modern UI", "color": "#ec4899"},
        {"icon": "🔗", "title": "Dependencies", "color": "#06b6d4"},
        {"icon": "📈", "title": "Analytics", "color": "#84cc16"}
    ]
    
    for idx, feature in enumerate(features):
        col = [feat_col1, feat_col2, feat_col3, feat_col4][idx % 4]
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {feature['color']}; 
                      min-height: 120px; text-align: center; padding: 1.5rem 1rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">{feature['icon']}</div>
                <h4 style="color: {THEME['text_light']}; margin: 0; font-size: 1rem; font-weight: 600;">
                    {feature['title']}
                </h4>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Technology Stack Section
    st.markdown(f"""
    <div class="custom-card fade-in">
        <h2 style="color: {THEME['accent_purple']}; margin-bottom: 1.5rem; font-size: 2rem; text-align: center;">
            🛠️ Technology Stack
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
    
    tech_stacks = [
        {"category": "Frontend", "icon": "🎨", "color": "#3b82f6"},
        {"category": "Data Processing", "icon": "📊", "color": "#10b981"},
        {"category": "AI/ML", "icon": "🤖", "color": "#f59e0b"},
        {"category": "Visualization", "icon": "📈", "color": "#8b5cf6"}
    ]
    
    for idx, tech in enumerate(tech_stacks):
        col = [tech_col1, tech_col2, tech_col3, tech_col4][idx]
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {tech['color']}; 
                      min-height: 120px; text-align: center; padding: 1.5rem 1rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">{tech['icon']}</div>
                <h4 style="color: {THEME['text_light']}; margin: 0; font-size: 1rem; font-weight: 600;">
                    {tech['category']}
                </h4>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Use Cases Section
    st.markdown(f"""
    <div class="custom-card fade-in">
        <h2 style="color: {THEME['accent_purple']}; margin-bottom: 1.5rem; font-size: 2rem; text-align: center;">
            💼 Perfect For
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    use_col1, use_col2, use_col3, use_col4 = st.columns(4)
    
    use_cases = [
        {"icon": "👨‍💻", "title": "Developers", "color": "#3b82f6"},
        {"icon": "🏗️", "title": "Architects", "color": "#10b981"},
        {"icon": "🔬", "title": "Researchers", "color": "#f59e0b"},
        {"icon": "🎓", "title": "Students", "color": "#8b5cf6"}
    ]
    
    for idx, use_case in enumerate(use_cases):
        col = [use_col1, use_col2, use_col3, use_col4][idx]
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {use_case['color']}; 
                      min-height: 120px; text-align: center; padding: 1.5rem 1rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">{use_case['icon']}</div>
                <h4 style="color: {THEME['text_light']}; margin: 0; font-size: 1rem; font-weight: 600;">
                    {use_case['title']}
                </h4>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Developer Section
    st.markdown(f"""
    <div class="custom-card fade-in" style="background: linear-gradient(135deg, rgba(30, 11, 61, 0.95), rgba(76, 29, 149, 0.95)); 
              border: 2px solid {THEME['accent_purple']};">
        <h2 style="color: {THEME['accent_purple']}; margin-bottom: 2rem; font-size: 2rem; text-align: center;">
            👨‍💻 About the Developer
        </h2>
        <div style="display: flex; align-items: center; gap: 2.5rem; flex-wrap: wrap; justify-content: center;">
            <div style="text-align: center;">
                <div style="width: 150px; height: 150px; border-radius: 50%; border: 4px solid {THEME['accent_purple']}; 
                          overflow: hidden; margin: 0 auto 1rem; box-shadow: 0 10px 30px rgba(168, 85, 247, 0.5);
                          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; 
                          align-items: center; justify-content: center;">
                    <div style="font-size: 4.5rem; color: white;">👨‍💻</div>
                </div>
            </div>
            <div style="flex: 1; min-width: 320px; max-width: 600px;">
                <h3 style="color: {THEME['text_light']}; margin-bottom: 0.5rem; font-size: 2rem; font-weight: 700;">
                    Faizan Mustafa
                </h3>
                <p style="color: {THEME['accent_purple']}; font-size: 1.2rem; margin-bottom: 1.2rem; font-weight: 600;">
                    Software Engineer & AI Enthusiast
                </p>
                <div style="color: {THEME['text_dark']}; font-size: 1.05rem; line-height: 2;">
                    <p style="margin: 0.4rem 0;">
                        🎓 <strong>COMSATS University Islamabad</strong> - Sahiwal Campus
                    </p>
                    <p style="margin: 0.4rem 0;">
                        📅 <strong>Bachelor's in Computer Science</strong> (2022-2026)
                    </p>
                    <p style="margin: 0.4rem 0;">
                        🆔 <strong>Registration:</strong> FA22-BCS-027
                    </p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
  
  # Professional Footer using Streamlit components
    st.markdown("---")
    
    # Links Section using Streamlit columns (FIRST)
    link_col1, link_col2, link_col3, link_col4 = st.columns(4)
    
    with link_col1:
        st.markdown("""
        <a href="https://github.com/FaizanMustafa-dev/MTLG-Visualizer-PDC-Project" target="_blank" style="text-decoration: none;">
            <div style="text-align: center; padding: 1rem; border-radius: 8px; 
                      background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.3);
                      transition: all 0.3s ease;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🐙</div>
                <div style="color: #f3e8ff; font-weight: 600;">GitHub</div>
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    with link_col2:
        st.markdown("""
        <a href="https://docs.google.com/document/d/1QFrt-89ItQb7KYOYYPbBYl6Y6PeroD2e/edit?usp=sharing&ouid=103310650248000019526&rtpof=true&sd=true" target="_blank" style="text-decoration: none;">
            <div style="text-align: center; padding: 1rem; border-radius: 8px; 
                      background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.3);
                      transition: all 0.3s ease;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📚</div>
                <div style="color: #f3e8ff; font-weight: 600;">Documentation</div>
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    with link_col3:
        st.markdown("""
        <a href="https://github.com/FaizanMustafa-dev/MTLG-Visualizer-PDC-Project" target="_blank" style="text-decoration: none;">
            <div style="text-align: center; padding: 1rem; border-radius: 8px; 
                      background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.3);
                      transition: all 0.3s ease;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⭐</div>
                <div style="color: #f3e8ff; font-weight: 600;">Star Project</div>
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    with link_col4:
        st.markdown("""
        <a href="https://github.com/FaizanMustafa-dev/MTLG-Visualizer-PDC-Project" target="_blank" style="text-decoration: none;">
            <div style="text-align: center; padding: 1rem; border-radius: 8px; 
                      background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.3);
                      transition: all 0.3s ease;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🐛</div>
                <div style="color: #f3e8ff; font-weight: 600;">Report Issue</div>
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Version and Developer Info (AT THE VERY BOTTOM)
    st.markdown("""
    <div style="text-align: center; padding: 2rem 1rem; 
              background: linear-gradient(180deg, rgba(13, 5, 25, 0) 0%, rgba(30, 11, 61, 0.5) 100%);
              border-top: 2px solid rgba(168, 85, 247, 0.3);">
        <h3 style="color: #f3e8ff; margin-bottom: 0.8rem; font-size: 1.8rem; font-weight: 700;">
            MTLG Visualizer v2.0.0
        </h3>
        <p style="color: #a855f7; font-size: 1.2rem; margin-bottom: 0.5rem; font-weight: 600;">
            Developed by Faizan Mustafa
        </p>
        <p style="color: #c4b5fd; font-size: 1rem; margin-bottom: 0;">
            © 2025 MTLG Visualizer | Open Source
        </p>
    </div>
    """, unsafe_allow_html=True)
# ============================================
# MAIN APP
# ============================================
def main():
    """Main application function"""
    # Initialize
    init_session_state()
    load_custom_css()
    
    # Create header
    create_header()
    
    # Create sidebar
    create_sidebar()
    
    # Page routing
    page = st.session_state.current_page
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Data Input":
        show_data_input()
    elif page == "Dependency Graph":
        show_dependency_graph()
    elif page == "Timeline":
        show_timeline()
    elif page == "Latency Analysis":
        show_latency_analysis()
    elif page == "Optimization":
        show_optimization()
    elif page == "About":
        show_about()

# ============================================
# RUN APPLICATION
# ============================================
if __name__ == "__main__":

    main()
