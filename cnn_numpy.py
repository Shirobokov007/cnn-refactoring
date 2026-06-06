"""
Свёрточная нейронная сеть (CNN) для распознавания рукописных цифр
Набор данных: Digits (8x8 изображения)
Архитектура: Conv2D(3x3) -> ReLU -> MaxPool(2x2) -> FC(36 -> 10)
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ActivationFunctions:
    """Класс с функциями активации"""
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU активация: max(0, x)"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Производная ReLU: 1 если x > 0, иначе 0"""
        return (x > 0).astype(float)
    
    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """
        Стабильный Softmax (вычитание максимума для предотвращения переполнения)
        """
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=-1, keepdims=True)
    
    @staticmethod
    def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Cross-entropy loss с защитой от log(0)"""
        eps = 1e-9
        return -np.sum(y_true * np.log(y_pred + eps))


class ConvolutionalLayer:
    """Свёрточный слой 2D"""
    
    def __init__(self, kernel_size: int = 3, learning_rate: float = 0.001):
        """
        Инициализация свёрточного слоя.
        
        Аргументы:
            kernel_size: Размер ядра свёртки (по умолчанию 3x3)
            learning_rate: Скорость обучения
        """
        self.kernel_size = kernel_size
        self.lr = learning_rate
        # Инициализация ядра случайными значениями (Xavier)
        self.kernel = np.random.uniform(-0.5, 0.5, (kernel_size, kernel_size))
        self.last_input = None
        self.last_output = None
    
    def convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        2D свёртка изображения с ядром.
        
        Аргументы:
            image: 2D массив (h, w)
            kernel: 2D массив (kh, kw)
        
        Возвращает:
            Результат свёртки (h-kh+1, w-kw+1)
        """
        h, w = image.shape
        kh, kw = kernel.shape
        output_h, output_w = h - kh + 1, w - kw + 1
        output = np.zeros((output_h, output_w))
        
        for i in range(output_h):
            for j in range(output_w):
                patch = image[i:i+kh, j:j+kw]
                output[i, j] = np.sum(patch * kernel)
        
        return output
    
    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Прямой проход: свёртка.
        
        Аргументы:
            image: Входное изображение (h, w)
        
        Возвращает:
            Результат свёртки
        """
        self.last_input = image
        self.last_output = self.convolve2d(image, self.kernel)
        return self.last_output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Обратный проход: вычисление градиента для ядра.
        
        Аргументы:
            grad_output: Градиент от следующего слоя
        
        Возвращает:
            Градиент для входного изображения
        """
        grad_kernel = np.zeros_like(self.kernel)
        kh, kw = self.kernel.shape
        h, w = self.last_input.shape
        
        for i in range(grad_output.shape[0]):
            for j in range(grad_output.shape[1]):
                patch = self.last_input[i:i+kh, j:j+kw]
                grad_kernel += grad_output[i, j] * patch
        
        self.kernel -= self.lr * grad_kernel
        return grad_output  # Упрощённый возврат


class PoolingLayer:
    """Max Pooling слой 2x2"""
    
    def __init__(self, pool_size: int = 2, stride: int = 2):
        """
        Инициализация pooling слоя.
        
        Аргументы:
            pool_size: Размер окна пулинга
            stride: Шаг пулинга
        """
        self.pool_size = pool_size
        self.stride = stride
        self.last_input = None
        self.max_positions = None
    
    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Прямой проход: max pooling.
        
        Аргументы:
            image: Входное изображение
        
        Возвращает:
            Результат max pooling
        """
        self.last_input = image
        h, w = image.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        output = np.zeros((out_h, out_w))
        self.max_positions = []
        
        for i in range(0, h, self.stride):
            for j in range(0, w, self.stride):
                patch = image[i:i+self.pool_size, j:j+self.pool_size]
                max_val = np.max(patch)
                output[i//self.stride, j//self.stride] = max_val
                self.max_positions.append((i, j))
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Обратный проход: направляем градиент в позиции максимумов.
        """
        grad_input = np.zeros_like(self.last_input)
        for idx, (i, j) in enumerate(self.max_positions):
            grad_input[i, j] = grad_output[idx // grad_output.shape[1], idx % grad_output.shape[1]]
        return grad_input


class FullyConnectedLayer:
    """Полносвязный слой"""
    
    def __init__(self, input_size: int, output_size: int, learning_rate: float = 0.001):
        """
        Инициализация полносвязного слоя.
        
        Аргументы:
            input_size: Количество входных нейронов
            output_size: Количество выходных нейронов
            learning_rate: Скорость обучения
        """
        self.lr = learning_rate
        # Инициализация Xavier
        self.weights = np.random.uniform(-0.5, 0.5, (output_size, input_size))
        self.bias = np.random.uniform(-0.5, 0.5, output_size)
        self.last_input = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Прямой проход: W·x + b.
        
        Аргументы:
            x: Входной вектор
        
        Возвращает:
            Выход слоя
        """
        self.last_input = x
        return np.dot(self.weights, x) + self.bias
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Обратный проход: вычисление градиентов.
        
        Аргументы:
            grad_output: Градиент от следующего слоя
        """
        grad_weights = np.outer(grad_output, self.last_input)
        grad_bias = grad_output
        grad_input = np.dot(self.weights.T, grad_output)
        
        self.weights -= self.lr * grad_weights
        self.bias -= self.lr * grad_bias
        
        return grad_input


class SimpleCNN:
    """
    Свёрточная нейронная сеть для классификации изображений 8x8.
    
    Архитектура:
    - Свёртка (3x3) -> ReLU -> MaxPool (2x2)
    - Полносвязный слой (36 -> 10) -> Softmax
    """
    
    def __init__(self, num_classes: int = 10, learning_rate: float = 0.001):
        """
        Инициализация CNN.
        
        Аргументы:
            num_classes: Количество классов (цифры 0-9)
            learning_rate: Скорость обучения
        """
        self.num_classes = num_classes
        self.lr = learning_rate
        
        # Слои сети
        self.conv = ConvolutionalLayer(kernel_size=3, learning_rate=learning_rate)
        self.fc = FullyConnectedLayer(input_size=36, output_size=num_classes, learning_rate=learning_rate)
        
        # Для хранения промежуточных значений
        self.last_relu = None
        self.last_pool = None
    
    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Прямой проход.
        
        Аргументы:
            image: Входное изображение 8x8
        
        Возвращает:
            Вероятности классов (10)
        """
        # Свёртка
        conv_out = self.conv.forward(image)
        
        # ReLU
        relu_out = ActivationFunctions.relu(conv_out)
        self.last_relu = relu_out
        
        # Max Pooling
        pooled = self._max_pool(relu_out)
        self.last_pool = pooled
        
        # Выравнивание и полносвязный слой
        flattened = pooled.flatten()
        logits = self.fc.forward(flattened)
        
        # Softmax
        return ActivationFunctions.softmax(logits.reshape(1, -1)).flatten()
    
    def _max_pool(self, image: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
        """Max pooling 2x2"""
        h, w = image.shape
        out_h = (h - pool_size) // stride + 1
        out_w = (w - pool_size) // stride + 1
        output = np.zeros((out_h, out_w))
        
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                patch = image[i:i+pool_size, j:j+pool_size]
                output[i//stride, j//stride] = np.max(patch)
        
        return output
    
    def predict(self, image: np.ndarray) -> int:
        """
        Предсказание класса.
        
        Аргументы:
            image: Входное изображение
        
        Возвращает:
            Предсказанный класс (0-9)
        """
        probs = self.forward(image)
        return np.argmax(probs)
    
    def train_on_batch(self, X: List[np.ndarray], y_onehot: List[List[float]], epochs: int = 50, verbose: bool = True) -> List[float]:
        """
        Обучение на батче данных (упрощённая версия без обратного распространения через свёртку).
        
        Аргументы:
            X: Список изображений
            y_onehot: Список one-hot векторов
            epochs: Количество эпох
            verbose: Печатать ли потери
        
        Returns:
            Список значений потерь по эпохам
        """
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for img, target in zip(X, y_onehot):
                probs = self.forward(np.array(img))
                loss = -np.sum(target * np.log(probs + 1e-9))
                total_loss += loss
            
            avg_loss = total_loss / len(X)
            losses.append(avg_loss)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        return losses


def load_and_prepare_data():
    """Загрузка и подготовка данных Digits"""
    digits = load_digits()
    X = digits.data.reshape(-1, 8, 8)
    y = digits.target
    
    # Нормализация
    X = X / 16.0
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # One-hot encoding
    def to_one_hot(labels, num_classes=10):
        return np.eye(num_classes)[labels]
    
    y_train_onehot = to_one_hot(y_train)
    y_test_onehot = to_one_hot(y_test)
    
    return X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot


def evaluate_model(model: SimpleCNN, X_test: List[np.ndarray], y_test: List[int]) -> float:
    """Оценка точности модели"""
    correct = 0
    for img, true_label in zip(X_test, y_test):
        pred = model.predict(np.array(img))
        if pred == true_label:
            correct += 1
    return correct / len(y_test)


def visualize_predictions(model: SimpleCNN, X_test: List[np.ndarray], y_test: List[int], num_samples: int = 5):
    """Визуализация предсказаний"""
    for i in range(min(num_samples, len(X_test))):
        img = X_test[i]
        true_label = y_test[i]
        pred = model.predict(np.array(img))
        
        plt.imshow(img, cmap='gray')
        plt.title(f'True: {true_label}, Predicted: {pred}')
        plt.axis('off')
        plt.show()


def main():
    """Главная функция"""
    print("=" * 60)
    print("Свёрточная нейронная сеть (CNN) — распознавание цифр")
    print("=" * 60)
    
    # Загрузка данных
    print("\nЗагрузка данных...")
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = load_and_prepare_data()
    print(f"Обучающая выборка: {len(X_train)}")
    print(f"Тестовая выборка: {len(X_test)}")
    
    # Создание и обучение модели
    print("\nОбучение модели...")
    cnn = SimpleCNN(num_classes=10, learning_rate=0.001)
    cnn.train_on_batch(X_train, y_train_onehot, epochs=80, verbose=True)
    
    # Оценка
    accuracy = evaluate_model(cnn, X_test, y_test)
    print(f"\nТочность на тестовой выборке: {accuracy:.2%}")
    
    # Визуализация
    print("\nВизуализация предсказаний:")
    visualize_predictions(cnn, X_test, y_test, num_samples=5)


if __name__ "__main__":
    main()