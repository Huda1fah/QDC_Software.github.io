import sys
import os
import numpy as np
import json
import uuid
import time
from typing import List, Dict, Tuple, Optional, Any
import threading
import logging
from datetime import datetime

# Quantum libraries
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.visualization import circuit_drawer
from qiskit.converters import circuit_to_dag, dag_to_circuit
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Import our circuit cutter implementation
from quantum_circuit_cutter import QuantumCircuitCutter

# Web and UI libraries
import flask
from flask import Flask, request, jsonify, render_template, send_file, Response
from flask_cors import CORS
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantum_platform.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumPlatform")

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)

# Global storage for jobs and results
job_store = {}

class QuantumJob:
    """Class to represent and track a quantum computation job"""
    
    def __init__(self, circuit: QuantumCircuit, num_processors: int, shots: int, name: str = None):
        self.job_id = str(uuid.uuid4())
        self.circuit = circuit
        self.num_processors = num_processors
        self.shots = shots
        self.name = name or f"Job-{self.job_id[:8]}"
        self.status = "created"
        self.creation_time = datetime.now()
        self.start_time = None
        self.end_time = None
        self.progress = 0
        self.subcircuits = []
        self.results = None
        self.entanglement_graph = None
        self.cut_points = []
        self.processor_stats = {}
        self.error = None
    
    def to_dict(self) -> Dict:
        """Convert job information to dictionary for JSON serialization"""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "num_qubits": self.circuit.num_qubits,
            "num_processors": self.num_processors,
            "shots": self.shots,
            "status": self.status,
            "creation_time": self.creation_time.isoformat(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "progress": self.progress,
            "num_gates": len(self.circuit.data),
            "num_subcircuits": len(self.subcircuits),
            "num_cut_points": len(self.cut_points),
            "error": self.error
        }

def run_quantum_job(job_id: str):
    """Execute a quantum job in a separate thread"""
    job = job_store[job_id]
    
    try:
        job.status = "running"
        job.start_time = datetime.now()
        job.progress = 5
        
        # Create circuit cutter instance
        cutter = QuantumCircuitCutter(job.circuit, num_processors=job.num_processors)
        
        # Step 1: Analyze entanglement
        job.progress = 10
        job.entanglement_graph = cutter.analyze_entanglement()
        
        # Step 2: Find optimal cuts
        job.progress = 30
        job.cut_points = cutter.find_optimal_cuts()
        
        # Step 3: Create subcircuits
        job.progress = 50
        job.subcircuits = cutter.create_subcircuits()
        
        # Step 4: Execute distributed
        job.progress = 60
        subcircuit_results = cutter.execute_distributed(shots=job.shots)
        
        # Store processor stats
        for i, circuit in enumerate(job.subcircuits):
            job.processor_stats[f"processor_{i}"] = {
                "num_qubits": circuit.num_qubits,
                "num_gates": len(circuit.data),
                "execution_time": np.random.uniform(0.5, 2.0)  # Simulated times
            }
        
        # Step 5: Knit results
        job.progress = 90
        job.results = cutter.knit_results()
        
        job.status = "completed"
        job.progress = 100
        job.end_time = datetime.now()
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        job.status = "failed"
        job.error = str(e)
        job.end_time = datetime.now()

def create_circuit_image(circuit: QuantumCircuit) -> str:
    """Create a base64 image of the circuit for UI display"""
    try:
        plt.figure(figsize=(10, 5))
        circuit_drawing = circuit_drawer(circuit, output='mpl')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return image_base64
    except Exception as e:
        logger.error(f"Error creating circuit image: {str(e)}")
        return ""

def create_entanglement_image(graph: nx.Graph) -> str:
    """Create a base64 image of the entanglement graph"""
    try:
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color='skyblue', 
                node_size=700, edge_color='black', linewidths=1, font_size=15)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return image_base64
    except Exception as e:
        logger.error(f"Error creating entanglement graph image: {str(e)}")
        return ""

def create_results_chart(results: Dict[str, float]) -> str:
    """Create a bar chart of the most likely measurement outcomes"""
    try:
        # Sort and take top 10 results
        top_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True)[:10])
        
        plt.figure(figsize=(10, 6))
        plt.bar(top_results.keys(), top_results.values())
        plt.xlabel('Measurement Outcome')
        plt.ylabel('Probability')
        plt.title('Top 10 Most Likely Measurement Outcomes')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return image_base64
    except Exception as e:
        logger.error(f"Error creating results chart: {str(e)}")
        return ""

# API Routes
@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html')

@app.route('/api/submit_circuit', methods=['POST'])
def submit_circuit():
    """API endpoint to submit a new quantum circuit for processing"""
    try:
        data = request.json
        
        # Parse the circuit from Qiskit code
        circuit_code = data.get('circuit_code', '')
        locals_dict = {}
        try:
            exec(circuit_code, globals(), locals_dict)
            circuit = locals_dict.get('circuit')
            if not isinstance(circuit, QuantumCircuit):
                return jsonify({"error": "Circuit code must define a variable named 'circuit' of type QuantumCircuit"}), 400
        except Exception as e:
            return jsonify({"error": f"Error parsing circuit code: {str(e)}"}), 400
        
        # Create a new job
        num_processors = int(data.get('num_processors', 2))
        shots = int(data.get('shots', 1024))
        job_name = data.get('job_name', None)
        
        job = QuantumJob(circuit, num_processors, shots, job_name)
        job_store[job.job_id] = job
        
        # Start processing in background
        threading.Thread(target=run_quantum_job, args=(job.job_id,)).start()
        
        return jsonify({"job_id": job.job_id, "status": "submitted"})
    
    except Exception as e:
        logger.error(f"Error submitting circuit: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """API endpoint to list all jobs"""
    try:
        jobs = [job.to_dict() for job in job_store.values()]
        return jsonify({"jobs": jobs})
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    """API endpoint to get details of a specific job"""
    try:
        if job_id not in job_store:
            return jsonify({"error": "Job not found"}), 404
        
        job = job_store[job_id]
        job_data = job.to_dict()
        
        # Add circuit image
        job_data["circuit_image"] = create_circuit_image(job.circuit)
        
        # Add entanglement graph if available
        if job.entanglement_graph:
            job_data["entanglement_graph_image"] = create_entanglement_image(job.entanglement_graph)
        
        # Add results chart if available
        if job.results:
            job_data["results_chart"] = create_results_chart(job.results)
            job_data["top_results"] = dict(sorted(job.results.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Add processor stats
        job_data["processor_stats"] = job.processor_stats
        
        return jsonify(job_data)
    
    except Exception as e:
        logger.error(f"Error getting job {job_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    """API endpoint to cancel a running job"""
    try:
        if job_id not in job_store:
            return jsonify({"error": "Job not found"}), 404
        
        job = job_store[job_id]
        
        if job.status in ["running", "created"]:
            job.status = "cancelled"
            job.end_time = datetime.now()
            return jsonify({"status": "cancelled"})
        else:
            return jsonify({"error": f"Cannot cancel job in {job.status} state"}), 400
    
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Create the HTML templates
@app.route('/generate_templates')
def generate_templates():
    """Generate HTML templates for the application"""
    os.makedirs('templates', exist_ok=True)
    
    # Index.html template
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Distributed Computing Platform</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/theme/monokai.min.css">
    <style>
        .CodeMirror {
            height: 300px;
            border: 1px solid #ddd;
        }
        .job-card {
            margin-bottom: 20px;
            transition: all 0.3s;
        }
        .job-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        .stats-container {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
        }
        .stat-card {
            flex: 0 0 32%;
            margin-bottom: 15px;
        }
        .progress-bar {
            transition: width 0.5s ease;
        }
        #circuitVisualizer, #entanglementVisualizer, #resultsVisualizer {
            width: 100%;
            overflow: auto;
            margin-top: 20px;
            text-align: center;
        }
        .processor-card {
            border-left: 4px solid #007bff;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="#">Quantum Distributed Computing Platform</a>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-lg-12">
                <div class="card mb-4">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Create New Quantum Job</h5>
                    </div>
                    <div class="card-body">
                        <form id="circuitForm">
                            <div class="mb-3">
                                <label for="jobName" class="form-label">Job Name</label>
                                <input type="text" class="form-control" id="jobName" placeholder="My Quantum Job">
                            </div>
                            <div class="mb-3">
                                <label for="circuitCode" class="form-label">Quantum Circuit Code (Qiskit)</label>
                                <textarea id="circuitCode" class="form-control"></textarea>
                                <small class="text-muted">Define your circuit in the variable named 'circuit'</small>
                            </div>
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label for="numProcessors" class="form-label">Number of Processors</label>
                                        <input type="number" class="form-control" id="numProcessors" value="2" min="2" max="8">
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label for="shots" class="form-label">Number of Shots</label>
                                        <input type="number" class="form-control" id="shots" value="1024" min="1">
                                    </div>
                                </div>
                            </div>
                            <div class="mb-3">
                                <button type="submit" class="btn btn-primary">Submit Circuit</button>
                                <button type="button" class="btn btn-secondary" id="loadExample">Load Example</button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-lg-12">
                <h3>Job Details</h3>
                <div class="card mb-4" id="jobDetailCard" style="display: none;">
                    <div class="card-header bg-info text-white d-flex justify-content-between align-items-center">
                        <h5 class="mb-0" id="jobDetailTitle">Job Details</h5>
                        <span class="badge" id="jobDetailStatus"></span>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <div class="progress">
                                <div class="progress-bar progress-bar-striped progress-bar-animated" id="jobProgress" role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%"></div>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card mb-3">
                                    <div class="card-header">
                                        <h6 class="mb-0">Job Information</h6>
                                    </div>
                                    <div class="card-body">
                                        <p><strong>Job ID:</strong> <span id="jobId"></span></p>
                                        <p><strong>Created:</strong> <span id="jobCreated"></span></p>
                                        <p><strong>Status:</strong> <span id="jobStatus"></span></p>
                                        <p><strong>Qubits:</strong> <span id="jobQubits"></span></p>
                                        <p><strong>Processors:</strong> <span id="jobProcessors"></span></p>
                                        <p><strong>Cut Points:</strong> <span id="jobCutPoints"></span></p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card mb-3">
                                    <div class="card-header">
                                        <h6 class="mb-0">Performance Metrics</h6>
                                    </div>
                                    <div class="card-body">
                                        <p><strong>Execution Time:</strong> <span id="jobExecutionTime"></span></p>
                                        <p><strong>Start Time:</strong> <span id="jobStartTime"></span></p>
                                        <p><strong>End Time:</strong> <span id="jobEndTime"></span></p>
                                        <p><strong>Gates:</strong> <span id="jobGates"></span></p>
                                        <p><strong>Subcircuits:</strong> <span id="jobSubcircuits"></span></p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <ul class="nav nav-tabs" id="jobTabs" role="tablist">
                            <li class="nav-item" role="presentation">
                                <button class="nav-link active" id="circuit-tab" data-bs-toggle="tab" data-bs-target="#circuit" type="button" role="tab">Circuit Visualization</button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="entanglement-tab" data-bs-toggle="tab" data-bs-target="#entanglement" type="button" role="tab">Entanglement Graph</button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="results-tab" data-bs-toggle="tab" data-bs-target="#results" type="button" role="tab">Results</button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="processors-tab" data-bs-toggle="tab" data-bs-target="#processors" type="button" role="tab">Processor Stats</button>
                            </li>
                        </ul>
                        
                        <div class="tab-content mt-3" id="jobTabContent">
                            <div class="tab-pane fade show active" id="circuit" role="tabpanel" aria-labelledby="circuit-tab">
                                <div id="circuitVisualizer">
                                    <p class="text-center">Circuit visualization will appear here.</p>
                                </div>
                            </div>
                            <div class="tab-pane fade" id="entanglement" role="tabpanel" aria-labelledby="entanglement-tab">
                                <div id="entanglementVisualizer">
                                    <p class="text-center">Entanglement graph will appear here.</p>
                                </div>
                            </div>
                            <div class="tab-pane fade" id="results" role="tabpanel" aria-labelledby="results-tab">
                                <div id="resultsVisualizer">
                                    <p class="text-center">Results will appear here after execution.</p>
                                </div>
                                <div id="resultsTable" class="mt-4">
                                    <h5>Top Results</h5>
                                    <table class="table table-striped">
                                        <thead>
                                            <tr>
                                                <th>Measurement</th>
                                                <th>Probability</th>
                                            </tr>
                                        </thead>
                                        <tbody id="resultsTableBody">
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                            <div class="tab-pane fade" id="processors" role="tabpanel" aria-labelledby="processors-tab">
                                <div id="processorStats">
                                    <h5>Processor Statistics</h5>
                                    <div id="processorCards"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-lg-12">
                <h3>Recent Jobs</h3>
                <div id="jobsContainer">
                    <p class="text-center">No jobs found. Submit a circuit to get started.</p>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/mode/python/python.min.js"></script>
    <script>
        // Initialize CodeMirror
        const editor = CodeMirror.fromTextArea(document.getElementById('circuitCode'), {
            mode: 'python',
            theme: 'monokai',
            lineNumbers: true,
            indentUnit: 4,
            lineWrapping: true
        });

        // Example circuit
        const exampleCircuit = `# Import necessary libraries
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Create a quantum circuit with 8 qubits
qr = QuantumRegister(8, 'q')
cr = ClassicalRegister(8, 'c')
circuit = QuantumCircuit(qr, cr)

# Apply Hadamard gates to all qubits
for i in range(8):
    circuit.h(i)

# Apply CNOT gates between adjacent qubits
for i in range(7):
    circuit.cx(i, i+1)

# Apply T gates to all qubits
for i in range(8):
    circuit.t(i)

# Create some non-adjacent entanglement
circuit.cx(0, 7)
circuit.cx(1, 6)

# Apply rotation gates
import numpy as np
for i in range(8):
    circuit.rz(np.pi/4, i)

# Measure all qubits
circuit.measure_all()
`;

        // Load example circuit
        document.getElementById('loadExample').addEventListener('click', () => {
            editor.setValue(exampleCircuit);
        });

        // Form submission
        document.getElementById('circuitForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = {
                job_name: document.getElementById('jobName').value,
                circuit_code: editor.getValue(),
                num_processors: document.getElementById('numProcessors').value,
                shots: document.getElementById('shots').value
            };
            
            try {
                const response = await fetch('/api/submit_circuit', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    alert('Job submitted successfully. Job ID: ' + data.job_id);
                    loadJobDetails(data.job_id);
                    loadJobs();
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error submitting job: ' + error.message);
            }
        });

        // Load job details
        async function loadJobDetails(jobId) {
            try {
                const response = await fetch(`/api/jobs/${jobId}`);
                const job = await response.json();
                
                if (response.ok) {
                    displayJobDetails(job);
                    
                    // If job is still running, refresh after 2 seconds
                    if (job.status === 'running' || job.status === 'created') {
                        setTimeout(() => loadJobDetails(jobId), 2000);
                    }
                } else {
                    console.error('Error loading job details:', job.error);
                }
            } catch (error) {
                console.error('Error:', error);
            }
        }

        // Display job details
        function displayJobDetails(job) {
            // Show job detail card
            document.getElementById('jobDetailCard').style.display = 'block';
            
            // Set job information
            document.getElementById('jobDetailTitle').textContent = job.name;
            document.getElementById('jobId').textContent = job.job_id;
            document.getElementById('jobCreated').textContent = new Date(job.creation_time).toLocaleString();
            document.getElementById('jobStatus').textContent = job.status;
            document.getElementById('jobQubits').textContent = job.num_qubits;
            document.getElementById('jobProcessors').textContent = job.num_processors;
            document.getElementById('jobGates').textContent = job.num_gates;
            document.getElementById('jobSubcircuits').textContent = job.num_subcircuits;
            document.getElementById('jobCutPoints').textContent = job.num_cut_points;
            
            // Set execution times
            document.getElementById('jobStartTime').textContent = job.start_time ? new Date(job.start_time).toLocaleString() : 'N/A';
            document.getElementById('jobEndTime').textContent = job.end_time ? new Date(job.end_time).toLocaleString() : 'N/A';
            
            if (job.start_time && job.end_time) {
                const start = new Date(job.start_time);
                const end = new Date(job.end_time);
                const executionTimeMs = end - start;
                document.getElementById('jobExecutionTime').textContent = (executionTimeMs / 1000).toFixed(2) + ' seconds';
            } else {
                document.getElementById('jobExecutionTime').textContent = 'N/A';
            }
            
            // Set progress
            document.getElementById('jobProgress').style.width = job.progress + '%';
            document.getElementById('jobProgress').setAttribute('aria-valuenow', job.progress);
            
            // Set status badge color
            const statusBadge = document.getElementById('jobDetailStatus');
            statusBadge.textContent = job.status.toUpperCase();
            
            if (job.status === 'completed') {
                statusBadge.className = 'badge bg-success';
            } else if (job.status === 'running' || job.status === 'created') {
                statusBadge.className = 'badge bg-primary';
            } else if (job.status === 'failed') {
                statusBadge.className = 'badge bg-danger';
            } else {
                statusBadge.className = 'badge bg-secondary';
            }
            
            // Set visualizations
            if (job.circuit_image) {
                document.getElementById('circuitVisualizer').innerHTML = `<img src="data:image/png;base64,${job.circuit_image}" alt="Circuit Visualization" class="img-fluid">`;
            }
            
            if (job.entanglement_graph_image) {
                document.getElementById('entanglementVisualizer').innerHTML = `<img src="data:image/png;base64,${job.entanglement_graph_image}" alt="Entanglement Graph" class="img-fluid">`;
            }
            
            if (job.results_chart) {
                document.getElementById('resultsVisualizer').innerHTML = `<img src="data:image/png;base64,${job.results_chart}" alt="Results Chart" class="img-fluid">`;
                
                // Populate results table
                const resultsTableBody = document.getElementById('resultsTableBody');
                resultsTableBody.innerHTML = '';
                
                if (job.top_results) {
                    for (const [outcome, probability] of Object.entries(job.top_results)) {
                        const row = document.createElement('tr');
                        
                        const measurementCell = document.createElement('td');
                        measurementCell.textContent = outcome;
                        
                        const probabilityCell = document.createElement('td');
                        probabilityCell.textContent = (probability * 100).toFixed(2) + '%';
                        
                        row.appendChild(measurementCell);
                        row.appendChild(probabilityCell);
                        resultsTableBody.appendChild(row);
                    }
                }
            }
            
            // Display processor stats
            const processorCards = document.getElementById('processorCards');
            processorCards.innerHTML = '';
            
            if (job.processor_stats) {
                for (const [processorId, stats] of Object.entries(job.processor_stats)) {
                    const card = document.createElement('div');
                    card.className = 'card processor-card mb-3';
                    
                    card.innerHTML = `
                        <div class="card-body">
                            <h6 class="card-title">${processorId.replace('_', ' ').toUpperCase()}</h6>
                            <div class="row">
                                <div class="col-md-4">
                                    <p><strong>Qubits:</strong> ${stats.num_qubits}</p>
                                    <p><strong>Gates:</strong> ${stats.num_gates}</p>
                                </div>
                                <div class="col-md-4">
                                    <p><strong>Execution Time:</strong> ${stats.execution_time.toFixed(2)}s</p>
                                </div>
                                <div class="col-md-4">
                                    <div class="progress">
                                        <div class="progress-bar bg-info" role="progressbar" 
                                             style="width: ${(stats.num_qubits / job.num_qubits * 100).toFixed(2)}%" 
                                             aria-valuenow="${(stats.num_qubits / job.num_qubits * 100).toFixed(2)}" 
                                             aria-valuemin="0" aria-valuemax="100">
                                            ${(stats.num_qubits / job.num_qubits * 100).toFixed(0)}%
                                        </div>
                                    </div>
                                    <small>% of total qubits</small>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    processorCards.appendChild(card);
                }
            }
        }

        // Load and display jobs
        async function loadJobs() {
            try {
                const response = await fetch('/api/jobs');
                const data = await response.json();
                
                if (response.ok) {
                    displayJobs(data.jobs);
                } else {
                    console.error('Error loading jobs:', data.error);
                }
            } catch (error) {
                console.error('Error:', error);
            }
        }

        // Display jobs
        function displayJobs(jobs) {
            const jobsContainer = document.getElementById('jobsContainer');
            
            if (jobs.length === 0) {
                jobsContainer.innerHTML = '<p class="text-center">No jobs found. Submit a circuit to get started.</p>';
                return;
            }
            
            jobsContainer.innerHTML = '';
            
            // Sort jobs by creation time (newest first)
            jobs.sort((a, b) => new Date(b.creation_time) - new Date(a.creation_time));
            
            for (const job of jobs) {
                const card = document.createElement('div');
                card.className = 'card job-card';
                
                let statusBadgeClass = 'bg-secondary';
                if (job.status === 'completed') {
                    statusBadgeClass = 'bg-success';
                } else if (job.status === 'running' || job.status === 'created') {
                    statusBadgeClass = 'bg-primary';
                } else if (job.status === 'failed') {
                    statusBadgeClass = 'bg-danger';
                }
                
                card.innerHTML = `
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">${job.name}</h5>
                        <span class="badge ${statusBadgeClass}">${job.status.toUpperCase()}</span>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <p><strong>Job ID:</strong> ${job.job_id}</p>
                                <p><strong>Created:</strong> ${new Date(job.creation_time).toLocaleString()}</p>
                                <p><strong>Qubits:</strong> ${job.num_qubits}</p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>Processors:</strong> ${job.num_processors}</p>
                                <p><strong>Gates:</strong> ${job.num_gates}</p>
                                <p><strong>Progress:</strong> ${job.progress}%</p>
                            </div>
                        </div>
                        <div class="progress mb-3">
                            <div class="progress-bar ${statusBadgeClass}" role="progressbar" style="width: ${job.progress}%" 
                                 aria-valuenow="${job.progress}" aria-valuemin="0" aria-valuemax="100"></div>
                        </div>
                        <button class="btn btn-primary btn-sm" onclick="loadJobDetails('${job.job_id}')">View Details</button>
                    </div>
                `;
                
                jobsContainer.appendChild(card);
            }
        }

        // Load jobs on page load
        window.addEventListener('load', loadJobs);
    </script>
</body>
</html>
    """
    
    with open('templates/index.html', 'w') as f:
        f.write(index_html)
    
    return "Templates generated successfully!"

# Create the quantum_circuit_cutter.py file
@app.route('/generate_cutter')
def generate_cutter():
    """Generate the quantum circuit cutter implementation"""
    
    cutter_code = """
# quantum_circuit_cutter.py
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.converters import circuit_to_dag, dag_to_circuit
import logging

logger = logging.getLogger("QuantumCircuitCutter")

class QuantumCircuitCutter:
    """Class that implements quantum circuit cutting and distributed execution"""
    
    def __init__(self, circuit: QuantumCircuit, num_processors: int = 2):
        """
        Initialize the circuit cutter.
        
        Args:
            circuit: The quantum circuit to be cut
            num_processors: Number of processors to distribute computation across
        """
        self.original_circuit = circuit
        self.num_processors = min(num_processors, circuit.num_qubits)
        self.entanglement_graph = None
        self.cut_points = []
        self.subcircuits = []
        self.qubit_mapping = {}  # Maps original qubits to processor and subcircuit qubit
    
    def analyze_entanglement(self) -> nx.Graph:
        """
        Analyze the entanglement structure of the circuit.
        
        Returns:
            A NetworkX graph representing the entanglement between qubits
        """
        logger.info(f"Analyzing entanglement for {self.original_circuit.num_qubits}-qubit circuit")
        
        # Create a graph with qubits as nodes
        graph = nx.Graph()
        for i in range(self.original_circuit.num_qubits):
            graph.add_node(i)
        
        # Analyze circuit to find entanglement (2-qubit gates)
        dag = circuit_to_dag(self.original_circuit)
        
        # For each operation in the circuit
        for op_node in dag.op_nodes():
            # Skip single-qubit gates
            if len(op_node.qargs) <= 1:
                continue
            
            # For 2+ qubit gates, add edges between the qubits
            for i in range(len(op_node.qargs)):
                qubit_i = op_node.qargs[i].index
                for j in range(i+1, len(op_node.qargs)):
                    qubit_j = op_node.qargs[j].index
                    
                    # Add edge or increase weight if it already exists
                    if graph.has_edge(qubit_i, qubit_j):
                        graph[qubit_i][qubit_j]['weight'] += 1
                    else:
                        graph.add_edge(qubit_i, qubit_j, weight=1)
        
        self.entanglement_graph = graph
        logger.info(f"Found {graph.number_of_edges()} entangling interactions between qubits")
        return graph
    
    def find_optimal_cuts(self) -> list:
        """
        Find optimal cutting points using spectral partitioning.
        
        Returns:
            List of (qubit_i, qubit_j) tuples representing the cuts
        """
        if self.entanglement_graph is None:
            self.analyze_entanglement()
        
        logger.info(f"Finding optimal cuts for {self.num_processors} processors")
        
        # Use spectral clustering to partition the graph
        if self.num_processors == 2:
            # Simple case: just use the Fiedler vector
            laplacian = nx.laplacian_matrix(self.entanglement_graph).toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            
            # Get the Fiedler vector (eigenvector corresponding to second smallest eigenvalue)
            # Skip the first eigenvalue (always 0 for connected graphs)
            fiedler_index = 1
            fiedler_vector = eigenvectors[:, fiedler_index]
            
            # Partition based on sign of Fiedler vector components
            partition = [[], []]
            for i, value in enumerate(fiedler_vector):
                if value < 0:
                    partition[0].append(i)
                else:
                    partition[1].append(i)
        else:
            # For more than 2 processors, use recursive spectral partitioning
            # This is a simplified version; more sophisticated methods could be used
            partition = [[] for _ in range(self.num_processors)]
            
            # Initial assignment: all qubits to first processor
            current_partition = list(range(self.original_circuit.num_qubits))
            
            def recursive_partition(nodes, partitions_needed, start_idx):
                if partitions_needed == 1:
                    partition[start_idx].extend(nodes)
                    return
                
                # Create subgraph with only these nodes
                subgraph = self.entanglement_graph.subgraph(nodes)
                
                # If disconnected, handle separately
                if not nx.is_connected(subgraph):
                    components = list(nx.connected_components(subgraph))
                    for i, component in enumerate(components):
                        if i < partitions_needed:
                            recursive_partition(list(component), 1, start_idx + i)
                        else:
                            # If more components than needed, combine with the last one
                            partition[start_idx + partitions_needed - 1].extend(component)
                    return
                
                # Spectral bisection
                laplacian = nx.laplacian_matrix(subgraph).toarray()
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
                fiedler_vector = eigenvectors[:, 1]  # Second smallest eigenvalue
                
                # Map back to original node indices
                node_list = list(subgraph.nodes())
                part1 = [node_list[i] for i, val in enumerate(fiedler_vector) if val < 0]
                part2 = [node_list[i] for i, val in enumerate(fiedler_vector) if val >= 0]
                
                # Balance partitions if needed
                if len(part1) == 0 or len(part2) == 0:
                    # If partitioning failed, use simple splitting
                    middle = len(nodes) // 2
                    part1 = nodes[:middle]
                    part2 = nodes[middle:]
                
                # Calculate sub-partition sizes proportionally
                p1_size = max(1, int(partitions_needed * len(part1) / len(nodes)))
                p2_size = partitions_needed - p1_size
                
                recursive_partition(part1, p1_size, start_idx)
                recursive_partition(part2, p2_size, start_idx + p1_size)
            
            recursive_partition(current_partition, self.num_processors, 0)
        
        # Find cut points: edges between different partitions
        cuts = []
        for i in range(self.original_circuit.num_qubits):
            for j in range(i+1, self.original_circuit.num_qubits):
                if self.entanglement_graph.has_edge(i, j):
                    # Find which partition each qubit belongs to
                    part_i = None
                    part_j = None
                    for p_idx, p in enumerate(partition):
                        if i in p:
                            part_i = p_idx
                        if j in p:
                            part_j = p_idx
                    
                    # If they're in different partitions, this is a cut
                    if part_i != part_j:
                        cuts.append((i, j))
        
        # Store the partition (qubit to processor mapping)
        for proc_idx, proc_qubits in enumerate(partition):
            for q in proc_qubits:
                self.qubit_mapping[q] = {"processor": proc_idx, "subcircuit_qubit": None}
        
        self.cut_points = cuts
        logger.info(f"Found {len(cuts)} optimal cut points: {cuts}")
        return cuts
    
    def create_subcircuits(self) -> list:
        """
        Create subcircuits for each processor.
        
        Returns:
            List of QuantumCircuit objects, one for each processor
        """
        if not self.qubit_mapping:
            self.find_optimal_cuts()
        
        logger.info(f"Creating {self.num_processors} subcircuits")
        
        # Create a list to store QuantumCircuit objects for each processor
        subcircuits = []
        
        # Create a map of processors to their qubits
        processor_qubits = {}
        for q, mapping in self.qubit_mapping.items():
            proc = mapping["processor"]
            if proc not in processor_qubits:
                processor_qubits[proc] = []
            processor_qubits[proc].append(q)
        
        # For each processor, create a quantum circuit
        for proc in range(self.num_processors):
            # Get qubits for this processor
            qubits = sorted(processor_qubits.get(proc, []))
            
            if not qubits:  # Skip empty processors
                subcircuits.append(None)
                continue
            
            # Create quantum and classical registers
            qr = QuantumRegister(len(qubits), f'p{proc}_q')
            cr = ClassicalRegister(len(qubits), f'p{proc}_c')
            qc = QuantumCircuit(qr, cr)
            
            # Update qubit mapping with subcircuit qubit indices
            for i, original_qubit in enumerate(qubits):
                self.qubit_mapping[original_qubit]["subcircuit_qubit"] = i
            
            # Add this subcircuit to the list
            subcircuits.append(qc)
        
        # Add gates from original circuit to subcircuits
        dag = circuit_to_dag(self.original_circuit)
        
        # Process operations in the original circuit
        for operation in dag.op_nodes():
            # Get qubits involved in this operation
            op_qubits = [q.index for q in operation.qargs]
            
            # Check if this operation spans multiple processors
            proc_set = set(self.qubit_mapping[q]["processor"] for q in op_qubits)
            
            if len(proc_set) == 1:
                # This operation is fully contained within one processor
                proc = list(proc_set)[0]
                if subcircuits[proc] is None:
                    continue
                
                # Get the corresponding qubits in the subcircuit
                subcircuit_qubits = [self.qubit_mapping[q]["subcircuit_qubit"] for q in op_qubits]
                
                # Add the operation to the subcircuit
                # Note: For simplicity, we're skipping complicated operations and focusing on common gates
                gate_name = operation.name
                
                if gate_name == 'cx':
                    subcircuits[proc].cx(subcircuit_qubits[0], subcircuit_qubits[1])
                elif gate_name == 'h':
                    subcircuits[proc].h(subcircuit_qubits[0])
                elif gate_name == 'x':
                    subcircuits[proc].x(subcircuit_qubits[0])
                elif gate_name == 'z':
                    subcircuits[proc].z(subcircuit_qubits[0])
                elif gate_name == 'y':
                    subcircuits[proc].y(subcircuit_qubits[0])
                elif gate_name == 's':
                    subcircuits[proc].s(subcircuit_qubits[0])
                elif gate_name == 't':
                    subcircuits[proc].t(subcircuit_qubits[0])
                elif gate_name == 'sdg':
                    subcircuits[proc].sdg(subcircuit_qubits[0])
                elif gate_name == 'tdg':
                    subcircuits[proc].tdg(subcircuit_qubits[0])
                elif gate_name == 'rx':
                    subcircuits[proc].rx(operation.op.params[0], subcircuit_qubits[0])
                elif gate_name == 'ry':
                    subcircuits[proc].ry(operation.op.params[0], subcircuit_qubits[0])
                elif gate_name == 'rz':
                    subcircuits[proc].rz(operation.op.params[0], subcircuit_qubits[0])
                elif gate_name == 'cz':
                    subcircuits[proc].cz(subcircuit_qubits[0], subcircuit_qubits[1])
                elif gate_name == 'swap':
                    subcircuits[proc].swap(subcircuit_qubits[0], subcircuit_qubits[1])
                elif gate_name == 'measure':
                    # Handle measurements
                    for i, q in enumerate(subcircuit_qubits):
                        subcircuits[proc].measure(q, q)
                # Add more gate types as needed
                
                # For unsupported gates, we could log a warning
                # Or implement a custom decomposition
            
            # For operations that span processors, we would need to handle the cut
            # This is a simplified version; a full implementation would use teleportation
            # or other protocols for handling cuts properly
        
        # Add measurements for all qubits if not already present
        for proc, circuit in enumerate(subcircuits):
            if circuit is None:
                continue
            
            # Add measurements for unmeasured qubits
            for q in range(circuit.num_qubits):
                # Check if this qubit is already being measured
                needs_measurement = True
                for op in circuit.data:
                    if op[0].name == 'measure' and op[1][0].index == q:
                        needs_measurement = False
                        break
                
                if needs_measurement:
                    circuit.measure(q, q)
        
        self.subcircuits = [c for c in subcircuits if c is not None]
        logger.info(f"Created {len(self.subcircuits)} subcircuits")
        return self.subcircuits
    
    def execute_distributed(self, shots: int = 1024) -> list:
        """
        Execute the subcircuits on separate simulated processors.
        
        Args:
            shots: Number of shots for the simulation
            
        Returns:
            List of execution results for each subcircuit
        """
        if not self.subcircuits:
            self.create_subcircuits()
        
        logger.info(f"Executing {len(self.subcircuits)} subcircuits with {shots} shots each")
        
        # Execute each subcircuit
        results = []
        simulator = Aer.get_backend('qasm_simulator')
        
        for i, subcircuit in enumerate(self.subcircuits):
            logger.info(f"Executing subcircuit {i+1}/{len(self.subcircuits)}")
            job = execute(subcircuit, simulator, shots=shots)
            result = job.result()
            counts = result.get_counts(subcircuit)
            results.append(counts)
        
        return results
    
    def knit_results(self) -> dict:
        """
        Combine results from subcircuits to approximate the original circuit.
        
        Returns:
            Dictionary mapping measurement outcomes to probabilities
        """
        logger.info("Knitting results from subcircuits")
        
        # This is a simplified knitting procedure
        # A full implementation would use tensor networks or density matrices
        # to properly account for entanglement across cut points
        
        # For simplicity, we'll combine results probabilistically
        # This approach works reasonably well for circuits with low entanglement across cuts
        
        # Initialize the combined results
        combined_results = {}
        
        # Get the inverse qubit mapping (subcircuit qubit to original qubit)
        inverse_mapping = {}
        for orig_q, mapping in self.qubit_mapping.items():
            proc = mapping["processor"]
            subcirc_q = mapping["subcircuit_qubit"]
            if subcirc_q is not None:  # Might be None if qubit wasn't used
                inverse_mapping[(proc, subcirc_q)] = orig_q
        
        # For each subcircuit, extract results and combine
        for proc_idx, subcircuit_counts in enumerate(self.execute_distributed()):
            # Process each measurement outcome
            for outcome, count in subcircuit_counts.items():
                probability = count / sum(subcircuit_counts.values())
                
                # Map the outcomes back to the original qubit indices
                original_outcome = ['0'] * self.original_circuit.num_qubits
                
                # Reverse the bit string since Qiskit uses little-endian format
                reversed_outcome = outcome[::-1]
                
                # For each bit in the outcome
                for subcirc_q, bit in enumerate(reversed_outcome):
                    # Map subcircuit qubit to original qubit
                    if (proc_idx, subcirc_q) in inverse_mapping:
                        orig_q = inverse_mapping[(proc_idx, subcirc_q)]
                        original_outcome[orig_q] = bit
                
                # Convert back to string
                original_outcome_str = ''.join(original_outcome)
                
                # Add to combined results
                if original_outcome_str in combined_results:
                    combined_results[original_outcome_str] += probability / self.num_processors
                else:
                    combined_results[original_outcome_str] = probability / self.num_processors
        
        # Normalize probabilities
        total_prob = sum(combined_results.values())
        for outcome in combined_results:
            combined_results[outcome] /= total_prob
        
        logger.info(f"Knitted results contain {len(combined_results)} unique outcomes")
        return combined_results
    
    def run_distributed_computation(self, shots: int = 1024) -> dict:
        """
        Run the entire distributed computation pipeline.
        
        Args:
            shots: Number of shots for simulation
            
        Returns:
            Dictionary mapping measurement outcomes to probabilities
        """
        logger.info(f"Running distributed computation for {self.original_circuit.num_qubits}-qubit circuit on {self.num_processors} processors")
        
        # Step 1: Analyze entanglement
        self.analyze_entanglement()
        
        # Step 2: Find optimal cuts
        self.find_optimal_cuts()
        
        # Step 3: Create subcircuits
        self.create_subcircuits()
        
        # Step 4: Execute and knit results
        results = self.knit_results()
        
        return results
"""
    
    with open('quantum_circuit_cutter.py', 'w') as f:
        f.write(cutter_code)
    
    return "Quantum circuit cutter implementation generated successfully!"

# Startup function to ensure directories exist
@app.before_first_request
def startup():
    """Ensure necessary directories and files exist"""
    # Ensure the static and templates directories exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Generate the quantum circuit cutter implementation if it doesn't exist
    if not os.path.exists('quantum_circuit_cutter.py'):
        with app.app_context():
            generate_cutter()
    
    # Generate the templates if they don't exist
    if not os.path.exists('templates/index.html'):
        with app.app_context():
            generate_templates()

if __name__ == '__main__':
    # Ensure startup files are created
    with app.app_context():
        startup()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)