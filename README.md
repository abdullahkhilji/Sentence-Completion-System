# Sentence-Completion-System
Task: To Design A Sentence Completion System.


The Task given to me was to Design a sentence completion system.
Given first 4 seed words of a sentence, it should predict the next word.
The training corpus used was from Sentiment140 (http://help.sentiment140.com/for-students/)


The name of the corpus file taken in the program is taken as taskData.csv , thus rename the file to taskData.csv on the 31st line of the code.
This should not be confused with vocab.csv that the program itself will gen- erate on line 134.
The program contains a total of 300 lines.



After training the data we should then compare it with the test data.
But since after so many refinements the data of size 1,00,000 reduces to mere 74,311 sentences having numerous bag’s of vector words that can be trusted to come from spoken english, the % further reduces after splitting into test and train sets’s.s Thus, for this task a sufficient size of data should be trained and this requires time.
The sample of 1,00,000 tweets gives an accuracy of 50-60% after some epochs. Here it is important to note that the output word is found by finding the vector closest to the output vector in the bag of vector of words.


Submitted on April 14 2018
