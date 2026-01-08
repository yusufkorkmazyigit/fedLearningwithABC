# ABC-Optimized Federated Averaging on PathMNIST

Bu proje, MedMNIST (PathMNIST) veri seti üzerinde **Yapay Arı Kolonisi (ABC)** algoritmasını kullanarak Federe Öğrenme (Federated Learning) performansını artırmayı hedefler.

## Senaryolar
Proje üç farklı senaryoyu karşılaştırır:
1. **Merkezi Öğrenme (Centralized):** Tüm verinin tek sunucuda olduğu ideal durum.
2. **Standart FedAvg:** Geleneksel federe ortalama yöntemi.
3. **ABC-Optimized FedAvg:** İstemci ağırlıklarının ABC algoritması ile dinamik olarak optimize edildiği önerilen yöntem.

## Kurulum
```bash
pip install -r requirements.txt