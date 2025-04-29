from ucimlrepo import fetch_ucirepo 
abalone = fetch_ucirepo(id=1)  
X = abalone.data.features 
y = abalone.data.targets 
