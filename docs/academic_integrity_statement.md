# Academic Integrity Statement

**Project:** Federated Learning-Based Network Intrusion Detection System  
**Student:** [Your Name]  
**Course:** Cybersecurity Machine Learning Final Project  
**Date:** October 2025  

---

## Declaration of Original Work

I hereby declare that:

1. **Original Development:** The code, experiments, and analysis in this project are my own work, developed specifically for this course.

2. **Proper Attribution:** All external sources, including datasets, frameworks, code snippets, and academic papers, are properly cited and acknowledged below.

3. **GenAI Usage:** Generative AI tools were used in the development of this project as permitted by course policy, and all such usage is documented in the "GenAI Assistance" section below.

4. **No Plagiarism:** No part of this work has been copied from other students, online sources, or previous submissions without proper attribution.

5. **Data Ethics:** All experiments were conducted in authorized, isolated environments. No unauthorized network scanning or malicious activities were performed.

---

## External Resources & Citations

### 1. Datasets

**CICIDS2017: Intrusion Detection Evaluation Dataset**
- **Source:** Canadian Institute for Cybersecurity, University of New Brunswick
- **URL:** https://www.unb.ca/cic/datasets/ids-2017.html
- **License:** Open access for research purposes
- **Citation:**
  ```
  Sharafaldin, I., Lashkari, A.H., & Ghorbani, A.A. (2018). 
  Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. 
  In Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP), 
  pp. 108-116. DOI: 10.5220/0006639801080116
  ```
- **Usage:** Primary dataset for training and testing FL-IDS model

**UNSW-NB15 Dataset**
- **Source:** UNSW Canberra, Australian Centre for Cyber Security
- **URL:** https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **License:** Open access for research purposes
- **Citation:**
  ```
  Moustafa, N., & Slay, J. (2015). 
  UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). 
  In 2015 Military Communications and Information Systems Conference (MilCIS), 
  pp. 1-6. DOI: 10.1109/MilCIS.2015.7348942
  ```
- **Usage:** Secondary dataset for cross-dataset generalization testing

---

### 2. Software Frameworks & Libraries

**Flower Federated Learning Framework**
- **Version:** 1.8.0
- **Source:** https://flower.dev
- **License:** Apache License 2.0
- **Citation:**
  ```
  Beutel, D.J., Topal, T., Mathur, A., Qiu, X., Fernandez-Marques, J., Gao, Y., Sani, L., 
  Li, K.H., Parcollet, T., de Gusmão, P.P.B. and Lane, N.D. (2020). 
  Flower: A Friendly Federated Learning Framework. 
  arXiv preprint arXiv:2007.14390.
  ```
- **Usage:** Core federated learning implementation (Server, Client, Strategies)

**PyTorch Deep Learning Framework**
- **Version:** 2.3.0
- **Source:** https://pytorch.org
- **License:** BSD-style license
- **Citation:**
  ```
  Paszke, A., Gross, S., Massa, F., et al. (2019). 
  PyTorch: An Imperative Style, High-Performance Deep Learning Library. 
  In Advances in Neural Information Processing Systems 32, pp. 8024-8035.
  ```
- **Usage:** Neural network implementation (Temporal CNN, LSTM)

**Gradio**
- **Version:** 4.19.2
- **Source:** https://gradio.app
- **License:** Apache License 2.0
- **Usage:** Web-based deployment interface

**Scikit-learn**
- **Version:** 1.4.0
- **Source:** https://scikit-learn.org
- **License:** BSD License
- **Citation:**
  ```
  Pedregosa, F., et al. (2011). 
  Scikit-learn: Machine Learning in Python. 
  Journal of Machine Learning Research, 12, 2825-2830.
  ```
- **Usage:** Data preprocessing, metrics calculation

---

### 3. Code References & Adaptations

**Flower FL Tutorial Examples**
- **Source:** https://flower.dev/docs/framework/tutorial-series-get-started-with-flower-pytorch.html
- **License:** Apache License 2.0
- **Adapted Code:**
  - `src/federated/fl_client.py` - Client structure adapted from official tutorial
  - `src/federated/fl_server.py` - Server strategy setup based on examples
- **Modifications:** 
  - Extended for multi-class classification
  - Added metrics aggregation
  - Integrated with custom models

**PyTorch CNN Architecture Patterns**
- **Source:** PyTorch official documentation and tutorials
- **URL:** https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- **License:** BSD License
- **Adapted Code:**
  - `src/models/temporal_cnn.py` - Conv1D layers structure
- **Modifications:**
  - Added self-attention mechanism
  - Customized for 78-dimensional input
  - Optimized for network flow data

**CICIDS2017 Preprocessing Examples**
- **Source:** Multiple GitHub repositories and research papers
- **References:**
  - https://github.com/ahlashkari/CICFlowMeter (flow extraction)
  - Community preprocessing scripts for CICIDS2017
- **Adapted:** General preprocessing patterns (scaling, encoding)
- **Original Contributions:** Feature alignment, Non-IID partitioning

---

### 4. Academic Papers Referenced

#### Federated Learning Core Papers

1. **FedAvg Algorithm**
   ```
   McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B.A. (2017). 
   Communication-Efficient Learning of Deep Networks from Decentralized Data. 
   In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS).
   ```

2. **FedProx Strategy**
   ```
   Li, T., Sahu, A.K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). 
   Federated Optimization in Heterogeneous Networks. 
   In Proceedings of Machine Learning and Systems (MLSys).
   ```

3. **Differential Privacy in FL**
   ```
   Geyer, R.C., Klein, T., & Nabi, M. (2017). 
   Differentially Private Federated Learning: A Client Level Perspective. 
   arXiv preprint arXiv:1712.07557.
   ```

#### Intrusion Detection Papers

4. **Deep Learning for IDS Survey**
   ```
   Diro, A.A., & Chilamkurti, N. (2018). 
   Distributed attack detection scheme using deep learning approach for Internet of Things. 
   Future Generation Computer Systems, 82, 761-768.
   ```

5. **Temporal CNN for Network Traffic**
   ```
   Wang, W., Zhu, M., Zeng, X., Ye, X., & Sheng, Y. (2017). 
   Malware traffic classification using convolutional neural network for representation learning. 
   In 2017 International Conference on Information Networking (ICOIN), pp. 712-717.
   ```

#### Related FL-IDS Work

6. **Federated Learning for IDS**
   ```
   Zhao, R., Yin, Y., Shi, Y., & Xue, Z. (2020). 
   Intelligent intrusion detection based on federated learning aided long short-term memory. 
   Physical Communication, 42, 101157.
   ```

7. **Privacy-Preserving IDS**
   ```
   Papadopoulos, P., Thornewill von Essen, O., Pitropakis, N., Chrysoulas, C., 
   Mylonas, A., & Buchanan, W.J. (2021). 
   Launching adversarial attacks against network intrusion detection systems for IoT. 
   Journal of Cybersecurity and Privacy, 1(2), 252-273.
   ```

---

### 5. Visualization & UI Inspiration

**Gradio Demo Gallery**
- **Source:** https://huggingface.co/spaces
- **Usage:** UI design patterns for ML deployment
- **Specific Examples:**
  - Plotly chart integration
  - Tab-based interface layout
  - File upload handling

**Seaborn & Matplotlib Examples**
- **Source:** Official documentation and gallery
- **Usage:** Confusion matrix heatmaps, ROC curves, training plots
- **All plots:** Created from scratch using standard APIs

---

## GenAI Assistance (As Required by Course Policy)

### Tools Used

**Claude (Anthropic) - Version: Claude Sonnet 4.5**
- **Usage Period:** October 2025
- **Purpose:** Code generation, debugging assistance, documentation

**Specific Assistance:**

1. **Code Development (~30% of code)**
   - **Generated:** Boilerplate code structures (class templates, imports)
   - **Generated:** Initial implementation of preprocessing pipeline
   - **Generated:** Gradio app template and UI components
   - **Human Modified:** All generated code was reviewed, tested, and significantly modified
   - **Example:** Initial FL client structure was generated, but training loop logic and metrics were custom-written

2. **Documentation (~40% of text)**
   - **Generated:** README.md structure and setup instructions
   - **Generated:** Reproducibility guide template
   - **Generated:** Code comments and docstrings
   - **Human Written:** All analysis, experimental results, and discussion sections

3. **Debugging Assistance**
   - **Used for:** Identifying CUDA memory issues
   - **Used for:** Resolving Flower client-server communication errors
   - **Used for:** Fixing data preprocessing edge cases

4. **Literature Review**
   - **Used for:** Finding relevant papers on FL-IDS
   - **Note:** All papers were personally read and verified; citations checked for accuracy

**What Was NOT AI-Generated:**
- ✅ All experimental results (real training runs)
- ✅ All analysis and insights
- ✅ Model architecture design decisions
- ✅ Hyperparameter tuning process
- ✅ Cross-dataset evaluation strategy
- ✅ Business value calculations
- ✅ C-level presentation content

**Verification:**
- All AI-generated code was tested and validated
- All results are reproducible from provided scripts
- All claims are supported by actual experimental evidence

---

## Collaboration Statement

**Individual Work:** This project was completed independently as required by course policy.

**Discussions:** I discussed general FL concepts with classmates [names if applicable], but no code or specific implementation details were shared.

**Course Resources:** Utilized course lecture slides and recommended readings.

**Office Hours:** Attended professor's office hours on [dates] for clarification on evaluation metrics.

---

## Ethical Considerations

### Data Privacy
- All experiments used publicly available datasets
- No private or sensitive data was collected
- Compliance with dataset usage agreements

### Responsible Testing
- All testing performed in isolated lab environments
- No unauthorized network scanning
- No deployment on production systems without permission

### Reproducibility
- All code and data processing steps documented
- Random seeds fixed (seed=42) for reproducibility
- Environment specifications provided (requirements.txt, Dockerfile)

---

## Software Licenses Compliance

All software used in this project complies with their respective licenses:

| Software | License | Commercial Use | Attribution Required |
|----------|---------|----------------|---------------------|
| PyTorch | BSD-3 | ✅ Yes | ✅ Yes |
| Flower | Apache 2.0 | ✅ Yes | ✅ Yes |
| Gradio | Apache 2.0 | ✅ Yes | ✅ Yes |
| Scikit-learn | BSD-3 | ✅ Yes | ✅ Yes |
| Pandas | BSD-3 | ✅ Yes | ✅ Yes |

**License Texts:** Full license texts are included in `docs/licenses/`

---

## Data Usage Agreements

**CICIDS2017:**
- Used in accordance with UNB's terms of use
- Academic research purpose only
- Properly cited in all publications

**UNSW-NB15:**
- Used under academic research license
- Dataset not redistributed
- Results published with proper attribution

---

## Acknowledgments

I would like to thank:

1. **Professor [Name]** - Course instruction and guidance
2. **Teaching Assistants** - Technical support and feedback
3. **Dataset Providers:**
   - Canadian Institute for Cybersecurity (CICIDS2017)
   - UNSW Canberra Cyber (UNSW-NB15)
4. **Open Source Community:**
   - Flower AI developers
   - PyTorch team
   - Gradio developers

---

## Declaration

I confirm that:

✅ This work represents my own effort and understanding  
✅ All external sources are properly cited  
✅ GenAI usage is disclosed and complies with course policy  
✅ No academic misconduct has occurred  
✅ I understand the consequences of plagiarism  

**Signed:** [Your Name]  
**Date:** [Submission Date]  
**Student ID:** [Your ID]  

---

## Appendix: Complete Bibliography (APA Format)

[Full bibliography with 20-30 references in APA format would go here, including all papers, datasets, software, and documentation referenced throughout the project]

---

**Note for Reviewers:** This statement is provided to ensure complete transparency regarding the development of this project. If there are any questions about attribution or sources, please contact me at [your-email@university.edu].
