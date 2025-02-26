# FlexDDM: Neural Network vs Traditional Method Comparison Framework

[FlexDDM](https://github.com/joyfan00/FlexDDM) is a Python framework for comparing the consistency between neural network methods and traditional methods in decision modeling. This framework extends the classic Diffusion Decision Model (DDM) and introduces neural network-based parameter estimation methods.

## Key Features

### Traditional Methods
- Standard Diffusion Decision Model (DDM)
- Diffusion Model for Conflict tasks (DMC)
- Dual-Stage Two-Process model (DSTP)
- Spotlight Shrinkage model (SSP)

### Neural Network Methods
- Neural-enhanced DDM (nDDMfz)
- Neural-enhanced DMC (nDMCfz)
- Neural-enhanced DSTP (nDSTPfz) 
- Neural-enhanced SSP (nSSPfz)

## Validation System

### Parameter Recovery Validation
- Validate parameter estimation accuracy through simulated data
- Compare parameter recovery effects between traditional and neural network methods

### Predictive Consistency Test
- Evaluate predictive consistency of different methods on real datasets
- Provide detailed model comparison metrics

### Computational Efficiency Analysis
- Compare computation time of different methods
- Provide optimization suggestions

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

View validation results:
- Parameter recovery plots: validation/parameter_validation/
- Model comparison plot: validation/model_validation.png
