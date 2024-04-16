# ViTrace


ViTrace is a state-of-the-art, dual-channel deep neural network designed to identify viral sequences within human transcriptomic data. It employs an integrated approach by combining Transformer and Convolutional Neural Network (CNN) architectures, maximizing their strengths to uncover viral signatures in human tumor sequencing data effectively. This tool is crucial for researchers focusing on the viral etiology of cancers.

For more details, visit the [ViTrace GitHub repository](https://github.com/Ying-Lab/ViTrace).


## Prerequisites

- Python 3.7


## Installation

To install ViTrace, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Ying-Lab/ViTrace
2. Navigate to the ViTrace directory:
   ```bash
   cd ViTrace
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
# Usage
## Analyzing Human Sequencing Data
To analyze human sequencing data, ensure that your sequencing read files are stored in the designated folder. Execute the following command:
   ```bash
   python predict.py  --folder <path>  
```
## Analyzing Mouse Sequencing Data

```
python predict_mouse.py
```
# Contributing
Contributions to TumorViTrap are welcome! Please refer to the contributing guidelines for more information on how to submit issues, fork the repository, and create pull requests.

# License

This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgments
We would like to thank all contributors and users of TumorViTrap for their support and feedback.
