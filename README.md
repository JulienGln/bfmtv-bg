# bfmtv-bg
Création d'un chatbot vérificateur de fake news.

Lien vers le sujet => [05-huggingface-transformers.ipynb](https://github.com/blancsw/deep_4_all/blob/main/cours/TP/05-huggingface-transformers.ipynb)

## Groupe
- Théo MATROD ([@C4RBON73](https://github.com/C4RBON73))
- Gor GRIGORYAN ([@gor-g](https://github.com/gor-g))
- Julien GALERNE ([@JulienGln](https://github.com/JulienGln))

## L'application
L'application se présente sous la forme d'une interface de chat textuel, similaire aux LLM actuels tesl que ChatGPT, Le Chat etc.

Pour cette interface, nous utilisons la librairie Python [**Gradio**](https://www.gradio.app/)

### Dataset
[FakeNews](https://huggingface.co/datasets/IJUN/FakeNews)

| **input** (*string*) | **instruct** (*string*) | **output** (*string*) |
|----------------------|-------------------------|------------------------|
| Facebook Messenger is eliminating Emoji's Facebook has announced that due to the success of emoji's and how widely recognized they have become[...]          | Identify whether the following news is real or fake. If it's real, answer 'real'; if it's fake, answer 'fake'. Wrap your answer with <end> and </end>. To answer the question, follow these steps internally: To improve the steps for identifying fake or real news, here’s a refined, clear, and concise approach based on the analysis of the provided text: --- ### **Steps to Identify Fake or Real News** 1. **Verify the Source**: - Check if the information comes from a credible, official, or well-known source (e.g., government agencies, reputable news[...]          | <end>fake</end> **Reasons:** 1. **Source Verification**: There is no credible or official announcement from Facebook or its representatives confirming the elimination of emojis from Facebook Messenger. 2. **Cross-Checking Facts**: The claim[...]             |
| Amazon giving away free budget phones. In a new move to pull market share away from[...]            | Identify whether the following news is real or fake. If it's real, answer 'real';[...]           | <end>fake</end> **Reasons:** 1. **Source Verification**: - There is no credible[...]             |
| Exemple 3            | Instruction 3           | Résultat 3             |

### Fine-tuning
### Déploiement

## Conclusion
