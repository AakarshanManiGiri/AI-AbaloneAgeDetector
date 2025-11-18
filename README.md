<p align="center">
  <img src="https://capsule-render.vercel.app/api?text=Abalone%20Age%20Prediction&type=waving&color=gradient&height=120"/>
</p>

<p align="center">
  <b>Linear Regression • Decision Trees • Random Forests</b><br>
  Machine learning workflow using the UCI Abalone dataset
</p>
<h2>Introduction</h2>

This project aims predicts the age of abalone using the UCI Abalone Dataset.
It employs a few models namely Linear Regression, Decision Trees, and Random Forests, and uses a modern ML pipeline with preprocessing, encoding, scaling, and cross-validation.
Even though it’s a simple dataset, the project demonstrates a clean E2E ML workflow.

<h2>Dataset</h2>
<b>Source:</b> UCI ML Repo
<b>Features:</b>
<ul>
  <li>Sex</li>
  <li>Length</li>
  <li>Diameter</li>
  <li>Height</li>
  <li>Whole Weight</li>
  <li>Shucked Weight</li>
  <li>Viscera Weight</li>
  <li>Shell Weight</li>
</ul>
<b>Target:</b> Age = No of Rings + 1.5

<h2>Models</h2>
<b>Linear Regression: </b>Linear Regression tries to percieve the relation between input and output by fitting a line that best predicts the target with the minimum amount of error.
<br><br>
<b>Decision Tree : </b>Decision Tree's make predictions oby splitting data repeatedly based on feature values. It chooses a feature and threshold that reduces prediction error the most and does this recursively to make a tree and makes its decision based on the average value in the leaf/node.
<br><br>
<b>Random Forest : </b>Random Forest functions similarly to decision tree but rather than constructing only one tree it produces multiple tree's which independantly make theyre own predictions and random forest returns the average of all predictions made.
<br><br>


