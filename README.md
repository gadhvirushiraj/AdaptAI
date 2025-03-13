# AdaptAI

_AdaptAI work is accepted at CHI 2025; Late-Breaking Work [[CHI Paper Link 🔗](https://programs.sigchi.org/chi/2025/program/content/19444)] | [[arXiv Paper Link 🔗](https://arxiv.org/abs/2503.09150)]

### **_A Personalized Solution to Sense Your Stress, Fix Your Mess, and Boost Productivity_**

![](./assets/adaptai-arch.png)

AdaptAI is a research project focused on AI-driven personalization to boosting workplace productivity and well-being. By integrating egocentric vision, audio, physiological signals, motion data and leveraging Large Language Models (LLMs) and Vision-Language Models (VLMs), AdaptAI offers:

- **Physical Well-being Support**: Timely interventions, such as movement reminders and micro-break suggestions.
- **Mental Well-being Assistance**: A Tone-Adaptive Conversational Agent (TCA) that provides tone supportive interactions during stress phases.
- **Task Automation**: AI agents handle small routine tasks like scheduling meetings and drafting emails.

## AdaptAI in Wild

<p align="center">
  <img src="./assets/adaptai-inwild.png" alt="AdaptAI example in wild" width="600"/>
</p>

## Further Details

### Model and API Keys

We utilized Open-Source Groq hosted models during the development and testing. During theses phases some models used were under Groq preview models category. To run AdaptAI, you need to provide your own Groq-API-Key.

### Hardware Specifications

- _Logitech C920 HD Pro Webcam_ (used for egocentric vision and audio)
- _MoveSense HR2_ (used for ECG, Heart Rate Monitoring and Motion Data)

## Citation

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


## Acknowledgement

A big thank you to everyone who took part in our user study and to those who helped create the demo video! A special thanks to Groq for their incredibly fast model inference times!
