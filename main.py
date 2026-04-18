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
        X = np.array(X)
        return np.dot(X,self.w) + self.b
    
    def compute_cost(self,y_pred,y_true):
        m = len(y_true)
        cost = (1/(2*m)) * np.sum((y_pred-y_true) ** 2)
        return cost
    
    def fit(self,X,y):
        X = np.array(X)
        y = np.array(y)

        m,n = X.shape

        self.w = np.zeros(n)
        self.b = 0.0

        for epoch in range(self.epoches):
            y_pred = self.predict(X)

            dw = (1/m) * np.dot(X.T,(y_pred - y))
            db = (1/m) * np.sum((y_pred-y))

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            cost = self.compute_cost(y_pred, y)
            self.costs.append(cost)

            if epoch % 100 == 0:
                print(f"Epoch:{epoch} - Cost: {self.compute_cost(y_pred,y)} - Params: {self.get_param()}")

    def get_param(self):
        return self.w,self.b

    def plot_reg(self, X, y):
        y_pred = self.predict(X)

        plt.scatter(y, y_pred, label="Pred vs Actual")
        plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label="Perfect Fit")

        plt.xlabel("Actual Y")
        plt.ylabel("Predicted Y")
        plt.title("Actual vs Predicted")
        plt.legend()
        plt.show()

    def plot_cost(self):

        plt.plot(self.costs)
        plt.xlabel("Epoches")
        plt.ylabel("Cost")
        plt.title("Cost Reduction")
        plt.show()


if __name__ == "__main__":

    # Features:
    # x1 = study hours
    # x2 = sleep hours

    X = [
        [1, 6],
        [2, 7],
        [3, 8],
        [4, 9],
        [5, 10]
    ]

    # Formula:
    # y = 2*x1 + 3*x2 + 5

    y = [25, 30, 35, 40, 45]

    model = LinearRegression(
        learning_rate=0.01,
        epoches=100
    )

    model.fit(X, y)

    print("\nFinal Parameters:")
    print(model.get_param())

    # Prediction
    sample = [[6, 11]]
    print("Prediction:", model.predict(sample))

    model.plot_cost()
    model.plot_reg(X,y)