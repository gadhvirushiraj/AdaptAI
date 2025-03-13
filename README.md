# AdaptAI

AdaptAI work is accepted at CHI 2025; Late-Breaking Work [[CHI Paper Link 🔗](https://programs.sigchi.org/chi/2025/program/content/19444)] [[arXiv Paper Link 🔗](https://arxiv.org/abs/2503.09150)]

### **_A Personalized Solution to Sense Your Stress, Fix Your Mess, and Boost Productivity_**

![](./assets/adaptai-arch.png)

AdaptAI is a research project focused on AI-driven personalization to boosting workplace productivity and well-being. By integrating egocentric vision, audio, physiological signals, motion data and leveraging Large Language Models (LLMs) and Vision-Language Models (VLMs), AdaptAI offers:

- **Physical Well-being Support**: Timely interventions, such as movement reminders and micro-break suggestions.
- **Mental Well-being Assistance**: A Tone-Adaptive Conversational Agent (TCA) that provides tone supportive interactions during stress phases.
- **Task Automation**: AI agents handle small routine tasks like scheduling meetings and drafting emails.

## AdaptAI in Wild 🎯

<p align="center">
  <img src="./assets/adaptai-inwild.png" alt="AdaptAI example in wild" width="600"/>
</p>
 

## Quick Start Guide 🚀 

### 1️⃣ Set Up Your Development Environment  

First, clone the repository and set up your virtual environment:  

```sh
git clone https://github.com/gadhvirushiraj/AdaptAI.git
cd AdaptAI
conda create --name adaptai-dev python=3.12
conda activate adaptai-dev
pip install -r requirements.txt
```

### 2️⃣ Configure API Keys & Email Credentials  

Before running the application, update the following variables in the code:  

- **Groq API Key**: Replace `GROQ_API_KEY` in **`main.py`** and **`tca.py`** with your Groq API key.  
- **Email Credentials**: Update `SMTP_USER` and `SMTP_PASS` inside **`main.py`** with your email and app password.  

### 3️⃣ Run AdaptAI  

- To **connect an ECG sensor** and log real-time data, run `movesense_sensor_data.py`.  
- To **enable interventions and audio agents**, run `main.py` (ensure egocentric vision and audio devices are connected and credentials updated).  
- To **launch the Tone-Adaptive Conversational Agent (TCA)**, use Streamlit:  

  ```sh
  streamlit run tca.py
  ```

  The UI will be available at [[http://localhost:8501](http://localhost:8501)]
  
## Further Details 📖

### Model and API Keys

We utilized Open-Source [Groq](https://groq.com/) hosted models during the development and testing. During theses phases some models used were under Groq [Preview Models](https://console.groq.com/docs/models#preview-models) category. To run AdaptAI, you need to provide your own `Groq-API-Key`.

### Hardware Specifications

- _Logitech C920 HD Pro Webcam_ (used for egocentric vision and audio)
- _MoveSense HR2_ (used for ECG, Heart Rate Monitoring and Motion Data)

## Citation 🏷️

Please cite the following if you reference our work:

```bibtex
@misc{gadhvi2025adaptaipersonalizedsolutionsense,
  title={AdaptAI: A Personalized Solution to Sense Your Stress, Fix Your Mess, and Boost Productivity},
  author={Rushiraj Gadhvi and Soham Petkar and Priyansh Desai and Shreyas Ramachandran and Siddharth Siddharth},
  year={2025},
  eprint={2503.09150},
  archivePrefix={arXiv},
  primaryClass={cs.HC},
  url={https://arxiv.org/abs/2503.09150},
}
```

## Acknowledgement 🤝

A big thank you to everyone who took part in our user study and to those who helped create the demo video! A special thanks to Groq for their incredibly fast model inference time!
