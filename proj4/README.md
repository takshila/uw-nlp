# Named-Entity Recognition with Structured Perceptrons

Named entities are phrases that contain the names of persons, organization, locations, etc.

NER as sequence tagging: Relation Extraction is the task of identifying & classifying the semantic relations between entities in text. The relation extraction task can be treated as a sequence labelling problem using BIO encoding, where the Beginning word of a named entity is marked with 'B', the following words of a named entity is marked with 'I' (i.e., inside) and words that are not part of an entity is marked with 'O' (i.e., outside). These labels are further marked with the types of named entities (e.g., ORG, PER, LOC). The performance of BIO & IO encoding schemes can be analyzed using conditional random field. The relation extraction task is treated as a structured sequence labelling problem where the application of the **Conditional Random Field** is most suitable.

**Structured Perceptrons** --> Classification with lots of features over structured models!

**Named-entity recognition (NER)** (also known as entity identification, entity chunking and entity extraction) is a subtask of information extraction that seeks to locate and classify named entities in text into pre-defined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc.

**Feature Templates**
* WordPair  --> Feature template made using previous word and current word.
* Word      --> Feature template made using current word.
* POS Tag   --> Feature template made using current parts of speech tag.
* Chunk Tag --> Feature template made using current chunk tag.
* Bigram NER Tag --> Feature template made using previous NER tag.
* NextWordPair --> Feature template using previous word and next word.
* Skip-gram --> Feature template using last to previous word and next to next word i.e. skipping previous and next word in the five word sequence.

Directly run hw5-structuredLearnings.py file in the same folder as that of ‘stories’ zip files and create a dev folder to store intermediate files.

OR

In order to run the .ipynb file, just run the command ‘jupiter notebook hw5-structuredLearnings.ipynb', which will open ipython notebook.

To RUN THE NOTEBOOK, go to cell and click on 'RUN ALL' which will run all the methods in the file.

Please place the .ipynb file in the same folder with that of ‘stories’ zip files and create a folder called dev to store intermediate files.

Also attached is the python file hw5-structuredLearnings.py generated from iPython notebook.

For the second part autoencoder.py is attached which has the information on the GRU cell. 

And autoencoder_rnnEncoder.py having RNN cell instead of BOW encoder.
