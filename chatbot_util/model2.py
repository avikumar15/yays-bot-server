from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Random forest classifier model with n_estimators = 100  

class RandomForestClassifier:   
  def model2(X, y):
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
     
    
    return classifier
