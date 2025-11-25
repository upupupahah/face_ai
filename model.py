import torch
import torch.nn as nn


class FaceDetector(nn.Module):
    """
    Улучшенная сверточная сеть для предсказания ограничивающего прямоугольника лица на изображении.

    Формат входа:
      - RGB изображение, ожидаемый размер: 256x256
    Формат выхода:
      - тензор размера (batch_size, 4) с координатами рамки в нормализованном виде: (cx, cy, w, h)
        где cx, cy — центр (в диапазоне [0,1] относительно ширины/высоты),
              w, h — ширина и высота (в диапазоне [0,1] относительно ширины/высоты).
    
    Улучшения:
      - BatchNorm для стабильности обучения
      - Больше фильтров (32/64/128/256) для лучшей представительности
      - Меньше сжатие в конце (8x8 вместо 4x4) для сохранения деталей локализации
      - Больше FC слоёв для лучшей регрессии
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Блок 1: 256x256 -> 128x128
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Блок 2: 128x128 -> 64x64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Блок 3: 64x64 -> 32x32
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Блок 4: 32x32 -> 16x16
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Блок 5: 16x16 -> 8x8 (сохраняем больше деталей, чем ранее)
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        # Более сложная FC часть для лучшей регрессии
        self.fc = nn.Sequential(
            nn.Flatten(),  # 256 * 8 * 8 = 16384
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid()  # нормализуем bbox в [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Проход вперёд: возвращает (batch, 4)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fc(x)
        return x