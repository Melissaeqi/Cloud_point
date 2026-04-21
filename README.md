1. **Какой baseline вы обучали**.

- exp_2
- best.pt
- обучала этот эксперемент на `cuda` не `mps`, так как решила расспаралелить обучение
- 
По эксперементам которые удалось произвести лучше оказался `PointNetLike_without_droupout` с обработкой данных FPS
```
def FPS(points_raw, N):
    
    N_raw = points_raw.shape[0]
    
    if N_raw < N:
        idx = np.random.choice(N_raw, N, replace = True)
        points = points_raw[idx]
        permut = np.random.permutation(N)
        points = points[permut]
    
    else:
        idx = np.zeros(N, dtype=int)
        idx[0] = np.random.randint(N_raw)

        diff = points_raw - points_raw[idx[0]]
        min_dist = np.sum(diff ** 2, axis=1)

        for i in range(1, N):

            idx[i] = np.argmax(min_dist)

            diff = points_raw - points_raw[idx[i]]
            dist = np.sum(diff ** 2, axis=1)
            min_dist = np.minimum(min_dist, dist)

        points = points_raw[idx]
        permut = np.random.permutation(N)
        points = points[permut]
    
    return points
        
```
В классе датасета надо прописать флаг `use_fps=True`

```
class PointNetLike_without_droupout(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP_1 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1),
        nn.BatchNorm1d(64),
        nn.ReLU()
        )
        self.MLP_2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.MLP_3 = nn.Sequential(
            nn.Conv1d(in_channels=1088, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=40, kernel_size=1)
        )
        
    def forward(self, x):
        x = x.transpose(1, 2) 
        
        x_1 = self.MLP_1(x)
        x_2 = self.MLP_2(x_1)
        
        h = torch.max(x_2, dim=2)[0]
        
        H = h.unsqueeze(2).repeat(1, 1, 2048)
        
        x_3 = torch.cat([x_1, H], dim=1) 
        
        x_4 = self.MLP_3(x_3)
        
        logits = torch.max(x_4, dim=2)[0] 
        
        return logits
        
```
2. **Какой N использовали**.
  
  N = 2048

3. **Какие аугментации включили**.
```
def rotate_z(points, degre=None):
    if degre is None:
        degre = np.random.uniform(0, 2 * np.pi)

    cos_a = np.cos(degre)
    sin_a = np.sin(degre)

    R = np.array([[cos_a, -sin_a, 0],[sin_a, cos_a, 0],[0, 0, 1]], dtype=np.float32)

    return points @ R.T
```
Поворачивает облако точек вокруг оси Z.
```
def jitter(points, sigma = 0.02, clip = 0.02):
    e = np.random.normal(0, sigma, points.shape)
    e = np.clip(e, -clip, clip)
    
    return  points + e
```
Добавляет случайный шум к точкам.
```
def train_transform(points):
    new_points = rotate_z(points)
    new_points = jitter(new_points)
    #point_dropout не буду делать, так как я не поняла, как его смысл,
    #нам же всё равно нужны данные размера N
    return new_points
```


4. **Какой optimizer / scheduler использовали.**

```
baseline_config = {
    "optimizer": "AdamW",
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "batch_size": 16, #стоит в лоадаре надо менять вручную
    "epochs": 75,
    "scheduler": "CosineAnnealingLR",
    "dropout": 0.05 #стоит в модели надо менять вручную
}
```
```
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=baseline_config["lr"],
    weight_decay=baseline_config["weight_decay"]
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=baseline_config["epochs"]
)

```
- 5. **Какая лучшая val_accuracy**.
  
   0.8460

- 6. **Какая итоговая test_accuracy (если test доступен)**
  
   0.8063
  
- 7. Какие 2–3 основных вывода вы сделали по экспериментам.
  
  Не всегда стоит сразу накидовать жесткую регуляризацию, иначе как и я будем долго грустить

  Иногда хорошая обработка данных даёт хороший прирост





