# bfmtv-bg
Création d'un chatbot vérificateur de fake news.

Lien vers le sujet => [05-huggingface-transformers.ipynb](https://github.com/blancsw/deep_4_all/blob/main/cours/TP/05-huggingface-transformers.ipynb)

## Groupe
Nom du groupe : BFMTV-bg

- Théo MATROD ([@C4RBON73](https://github.com/C4RBON73))
- Gor GRIGORYAN ([@gor-g](https://github.com/gor-g))
- Julien GALERNE ([@JulienGln](https://github.com/JulienGln))

## L'application
L'application se présente sous la forme d'une interface de chat textuel, similaire aux LLMs actuels tels que ChatGPT, Le Chat etc.

Pour cette interface, nous utilisons la librairie Python [**Gradio**](https://www.gradio.app/).
Le code de l'interface est disponible dans le fichier [app.py](./app.py) et utilise la version 2 de notre modèle.

Les requêtes sont formulées au chatbot en anglais.

### Dataset
#### Dataset 1
[FakeNews](https://huggingface.co/datasets/IJUN/FakeNews)

| **input** (*string*) | **instruct** (*string*) | **output** (*string*) |
|----------------------|-------------------------|------------------------|
| Facebook Messenger is eliminating Emoji's Facebook has announced that due to the success of emoji's and how widely recognized they have become[...]          | Identify whether the following news is real or fake. If it's real, answer 'real'; if it's fake, answer 'fake'. Wrap your answer with <end> and </end>. To answer the question, follow these steps internally: To improve the steps for identifying fake or real news, here’s a refined, clear, and concise approach based on the analysis of the provided text: --- ### **Steps to Identify Fake or Real News** 1. **Verify the Source**: - Check if the information comes from a credible, official, or well-known source (e.g., government agencies, reputable news[...]          | <end>fake</end> **Reasons:** 1. **Source Verification**: There is no credible or official announcement from Facebook or its representatives confirming the elimination of emojis from Facebook Messenger. 2. **Cross-Checking Facts**: The claim[...]             |
| Amazon giving away free budget phones. In a new move to pull market share away from[...]            | Identify whether the following news is real or fake. If it's real, answer 'real';[...]           | <end>fake</end> **Reasons:** 1. **Source Verification**: - There is no credible[...]             |
| Exemple 3            | Instruction 3           | Résultat 3             |

#### Dataset 2
[LargeFakeNews](https://huggingface.co/datasets/IJUN/LargeFakeNews)

Ce dataset est fait par le même auteur que le premier mais contient bien plus de lignes (plus de 5000 contre environ 300 pour le premier).

| **input** (*string*) | **instruct** (*string*) | **output** (*string*) |
|----------------------|-------------------------|-----------------------|
| HOT OR NOT? Are Trump Properties Selling Like Hotcakes? [Video] One agent Gilad Azaria has 7 listings in Trump buildings, and since the election he s had 4 serious buyers in the $3 million range. He also says 2 of his clients who already live in Trump properties want to trade up in the same buildings to more expensive places. The agents with whom we spoke say the Trump properties are top quality, and the brand is now stronger than ever. Celeb agents Josh and Matt Altman say it s a no-brainer living in a building owned by the president is awesome and a super good investment. Via: TMZ            | Classify the text as 'real' or 'fake' news. Wrap your answer with <end> and </end>.           | <end>fake</end> The text is likely fake news due to its reliance on anecdotal evidence, sensational language, and lack of credible sourcing. TMZ, an entertainment-focused outlet, is not a reliable source for real estate or political news. Additionally, the claims are not supported by independent data or verified by reputable news organizations. The promotional tone and potential conflicts of interest further undermine its credibility.             |
| (VIDEO) CONGRESSIONAL BUDGET OFFICE: DEBT HEADED TO 103% OF GDP – LAST SEEN AT END OF WWII CBO DIRECTOR: How long the nation could sustain such growth in federal debt is impossible to predict with any confidence. At some point, investors would begin to doubt the government s willingness or ability to meet its debt obligations, requiring it to pay much higher interest costs in order to continue borrowing money. Such a fiscal crisis would present policymakers with extremely difficult choices and would probably have a substantial negative impact on the country.            | Classify the text as 'real' or 'fake' news. Wrap your answer with <end> and </end>.           | The correct answer is <end>fake</end>. The content uses sensational language and lacks verifiable sources or corroborating evidence from official CBO reports or credible news outlets. Additionally, the claim about U.S. debt reaching 103% of GDP and comparisons to WWII levels are not supported by current economic data or expert analyses.             |
| Trump arrives in South Korea for talks on nukes, trade OSAN, South Korea (Reuters) - U.S. President Donald Trump landed in South Korea on Tuesday, the second leg of his 12-day Asia trip dominated by the North Korean nuclear standoff. South Koreans are bracing for the possibility that Trump s state visit could risk further inflaming tensions with North Korean leader Kim Jong Un, who has stepped up his pursuit of nuclear weapons that could soon be capable of striking the mainland United States. Trump will visit with U.S troops and is also expected to raise criticisms of a U.S-South Korean trade pact when he meets with President Moon Jae-in in Seoul.            | Classify the text as 'real' or 'fake' news. Wrap your answer with <end> and </end>.           | 	<end>real</end> **Reasons:** 1. **Source Verification**: The content is attributed to Reuters, a highly credible and well-established news agency. 2. **Cross-Checking Facts**: The details about Trump's visit to South Korea, his focus on North Korean nuclear tensions, and trade discussions align with widely reported events during his Asia trip in 2017. 3. **Tone and Language**: The language is neutral and factual, without sensationalism or emotional manipulation. 4. **Contextual Alignment**: The story fits within the broader context of Trump's foreign policy priorities and diplomatic engagements during his presidency. 5. **No Red Flags**: The article is professionally written, lacks grammatical errors, and does not use clickbait tactics. Based on these factors, the content is consistent with real news.             |

### Déploiement
Le projet est poussé sur un dépôt Hugging Face, sous 2 versions : 
- V1 : [Gor-bepis/fact-checker-bfmtg-v1](https://huggingface.co/Gor-bepis/fact-checker-bfmtg-v1)
- V2 : [Gor-bepis/fact-checker-bfmtg-v2](https://huggingface.co/Gor-bepis/fact-checker-bfmtg-v2)

Il est automatiquement poussé sur le dépôt via une cellule du notebook :
```py
model.push_to_hub("Gor-bepis/fact-checker-bfmtg-v2")
tokenizer.push_to_hub("Gor-bepis/fact-checker-bfmtg-v2")
```

L'API d'Inférence ne semble cependant pas marcher, même si le modèle est utilisable en local (cf. [app.py](./app.py)).

### Fine-tuning
Le modèle **V1** est ajusté (fine-tuning) à partir du dataset décrit [ci-dessus](#dataset-1).

Le modèle **V2** est ajusté (fine-tuning) à partir du dataset décrit [ci-dessus](#dataset-2). Nous avons baissé le learning rate à 0.0001 car le modèle avait tendance à sur-apprendre. Nous avons également augmenté le nombre d'étapes, de 1000 à 1500 pour améliorer ses performances.

Le code du fine-tuning pour les modèles sont disponibles dans les notebook :
- V1 : [bftmtv-bg-kaggle](./bfmtv-bg-kaggle.ipynb)
- V2 : [bfmtv-bg-kaggle-v2](./bfmtv-bg-kaggle-v2.ipynb)

## Conclusion
Nous avons mis en place un chatbot qui vérifie si la news qu'on lui donne est vraie ou fausse. 
Pour cela, il vérifie plusieurs points tels que le ton de l'info, la source ou même la cohérence de l'info (en vérifiant d'autres sources).
Le modèle a été fine-tuned à partir d'un dataset répertoriant des news, vraies ou fausses.

Enfin, nous avons choisi de prendre la **version 2** du modèle car elle est plus précise, malgré le fait que le bot ne termine pas toujours ses phrases.

Pour améliorer le modèle, il faudrait retravailler le jeu de données pour raccourcir les inputs et outputs, notamment pour que le bot puisse faire des phrases complètes.