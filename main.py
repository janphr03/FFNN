import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from sympy.codegen.cnodes import sizeof
from torchvision import datasets


class NeuralNetwork:
    def __init__(self, inputSize = 784, hiddenLayers = [512,512], outputSize = 10):
        self.inputSize = inputSize
        self.hiddenLayers = hiddenLayers
        self.outputSize = outputSize
        self.weights = []
        self.biases = []

        # input zu hidden Layers Matrix
        self.weights.append(0.01 * np.random.rand(inputSize, hiddenLayers[0]))
        self.biases.append(np.zeros((1, hiddenLayers[0])))

        # legt beide Verbindungs-Matrizen an
        for i in range(len(hiddenLayers) - 1):
            self.weights.append(0.01 * np.random.rand(hiddenLayers[i], hiddenLayers[i + 1]))
            self.biases.append(np.zeros((1, hiddenLayers[i + 1])))

        # hidden Layer zu Output Matrix erstellen
        self.weights.append(0.01 * np.random.rand(hiddenLayers[-1], outputSize))
        self.biases.append(np.zeros((1, outputSize)))



    # ReLU bringt "nichtlinearität" in das Netzwerk, damit komplexere Zusammenhänge erkannt werden können (einzelne Bereiche können separat definiert werden)
    def relu(self, x):
        return np.maximum(0, x)

    # Softmax wandelt Zahlen in Promille Werte um (für den Output notwendig)
    def softmax(self, x):
        # Subtrahiere den maximalen Wert pro Zeile, um numerische Stabilität zu gewährleisten
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


    def forwardPass(self, inputs):
        # Initialisiere eine Liste 'layers' mit dem Eingabevektor als erster Schicht
        layers = [inputs]

        # Iteriere über alle Gewichtsmatrizen im Netzwerk
        for i in range(len(self.weights)):
            # Berechne den gewichteten Input (z) für die aktuelle Schicht:
            # z = a_prev • W + b
            z = np.dot(layers[-1], self.weights[i]) + self.biases[i]

            # Wenn wir in der letzten Schicht sind → Softmax (für Klassifikation)
            if i == len(self.weights) - 1:
                a = self.softmax(z)
            else:
                # In allen vorherigen Schichten → ReLU als Aktivierungsfunktion
                a = self.relu(z)

            # Speichere das Ergebnis (a) als neue Aktivierungsschicht
            layers.append(a)

        # Gib die Ausgabe der letzten Schicht zurück
        return layers[-1]

    # Loss berechnen
    def crossEntropy(self, y_hat, y):
        """
        y_hat: Output aus Softmax, shape (1, 10)
        y: One-Hot-Vektor z.B. [0,0,0,1,0,0,0,0,0,0]
        """
        epsilon = 1e-12  # für numerische Stabilität, da log(0) NaN
        y_hat = np.clip(y_hat, epsilon, 1. - epsilon) # y_hat muss zwischen (0 - 1) liegen
        return -np.sum(y * np.log(y_hat))


    # Backpropagation

    # das ersetzt Sigmoid- oder andere veraltete Mathem. Funktionen
    # x ist das Skalarprodukt was abgeleitet wird
    def reluDerivative(self, x):

        return (x > 0).astype(float)
        # gibt 1 oder 0 zurück, da entweder 5x = x oder 5 ist 0 wenn man ableitet



    def backwardPass(self, x, y, learning_rate=0.01):
        # Forward Pass (inkl. Zwischenschritte speichern)
        activations = [x]
        zs = []

        a = x
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            zs.append(z)
            if i == len(self.weights) - 1:
                a = self.softmax(z)
            else:
                a = self.relu(z)
            activations.append(a)

        # Delta für Output-Schicht
        delta = activations[-1] - y  # y_hat - y
        dWs = []
        dbs = []

        # Rückwärts durch die Schichten
        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            z = zs[i]

            dW = np.dot(a_prev.T, delta)
            db = np.sum(delta, axis=0, keepdims=True)

            dWs.insert(0, dW)
            dbs.insert(0, db)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(zs[i - 1])

        # Parameter aktualisieren
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dWs[i]
            self.biases[i] -= learning_rate * dbs[i]

    def evaluate_and_show(self, mnist_data, index=None):
        if index is None:
            index = np.random.randint(0, len(mnist_data))

        img_tensor, true_label = mnist_data[index]
        img_np = img_tensor.numpy().reshape(1, 784)

        probs = self.forwardPass(img_np)
        prediction = np.argmax(probs, axis=1)[0]

        plt.imshow(img_tensor.squeeze(), cmap="gray")
        plt.title(f"Echtes Label: {true_label} – Vorhersage: {prediction}")
        plt.axis("off")
        plt.show()

        print(f" Wahrscheinlichkeiten: {np.round(probs, 3)}")
        print(f" Korrekt: {prediction == true_label}")

    def load_weights(self, state_dict):
        weight_keys = [k for k in state_dict if "weight" in k]
        bias_keys = [k for k in state_dict if "bias" in k]
        for i in range(len(weight_keys)):
            self.weights[i] = state_dict[weight_keys[i]].numpy().T
            self.biases[i] = state_dict[bias_keys[i]].numpy().reshape(1, -1)

###===================================================================




###===================================================================

# 1. Modell instanziieren
model = NeuralNetwork()

# 2. Gewichte aus PyTorch laden
state_dict = torch.load("mnist_model.pth", map_location=torch.device("cpu"))
model.load_weights(state_dict)

# 3. Testdaten laden (aus MNIST)
transform = transforms.ToTensor()
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# 4. Zufälliges Bild testen und Ergebnis anzeigen
model.evaluate_and_show(mnist_test)


for i, W in enumerate(model.weights):
    plt.figure(figsize=(6, 4))
    sns.heatmap(W, cmap="coolwarm", cbar=True)
    plt.title(f"Gewichtsmatrix {i} – Shape {W.shape}")
    plt.xlabel("Ausgangsneuronen")
    plt.ylabel("Eingangsneuronen")
    plt.tight_layout()
    plt.show()
#google Colab oder Jupyter
