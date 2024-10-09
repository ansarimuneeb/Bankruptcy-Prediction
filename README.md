Refer "Research Report.pdf" for an overall understanding of corporate bankruptcy and results of the research.

Refer "Technical Report.pdf" for rationale for technical decisions made throughout the project.

This project was succesfull at creating a model that predicts bankruptcy with 95.45% accuracy with the Support Vector Machines (SVM) algorithm. However there were certain limitations due to the nature of bankruptcy.

• Bankrupt cases were very overlapping with non-bankrupt cases, there was no clear distinction between the two. This reflects the unpredictable nature of bankruptcy itself.
• Even a high number of Principal Components were not able to explain up to 80% variation in the data. In simple terms this means that no matter how many new dimensions were created to explain the behavior bankruptcy, it was never sufficient.
•Even though we have a model that succesfully predicts 95.45% of the total bankrupt cases, our model is only correct 17.87% of the times when it is making a bankrupt prediction. This implies that our model is very quick to call out companies as bankrupt even when they are not. This is acceptable as it is better to falsely call a company as bankrupt rather than declaring a potential bankrupt company as safe.
•Interpretable Machine Learning (IML) methods revealed that financial ratios that seem to be correlated to bankruptcy are not directly causing it. This implies that there are other factors that are contributing towards causing bankruptcy which are not present in the dataset.
