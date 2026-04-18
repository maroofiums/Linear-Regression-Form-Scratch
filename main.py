import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01,epoches=1000):
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.w = None
        self.b = None

        self.costs = []
    
    def predict(self,X):
        return self.w * X + self.b
    
    def compute_cost(self,y_pred,y_true):
        m = len(y_true)
        cost = (1/(2*m)) * np.sum((y_pred-y_true) ** 2)
        return cost
    
    def fit(self,X,y):
        X = np.array(X)
        y = np.array(y)

        self.w = 0.0
        self.b = 0.0
        m = len(y)
        for epoch in range(self.epoches):
            y_pred = self.predict(X)

            dw = (1/m) * np.sum((y_pred-y)*X)
            db = (1/m) * np.sum((y_pred-y))

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            cost = self.compute_cost(y_pred, y)
            self.costs.append(cost)

            if epoch % 100 == 0:
                print(f"Epoch:{epoch} - Cost: {self.compute_cost(y_pred,y)} - Params: {self.get_param()}")

    def get_param(self):
        return self.w,self.b

    def plot_reg(self,X,y):
        X = np.array(X)
        y = np.array(y)

        plt.scatter(X,y,label="Data Points")
        plt.plot(X,self.predict(X),label="Regression Line")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Linear Regression")
        plt.legend()
        plt.show()

# Example Usage: 
if __name__ == "__main__":
    X = [1,2,3,4,5]
    y = [5,7,9,11,13]

    model = LinearRegression(learning_rate=0.1, epoches=10000)
    model.fit(X,y)

    print(model.get_param())
    print(model.predict(6))

    model.plot_reg(X,y)