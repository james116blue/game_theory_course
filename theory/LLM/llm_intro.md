---
marp: true
theme: default
paginate: true
style: |
  :root {
    --color-bg: #0f1117;
    --color-fg: #e8eaf6;
    --color-accent: #7c4dff;
    --color-accent2: #00e5ff;
    --color-muted: #90a4ae;
  }
  section {
    background: #0f1117;
    color: #e8eaf6;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 22px;
  }
  h1 { color: #7c4dff; font-size: 2em; border-bottom: 2px solid #7c4dff; padding-bottom: 8px; }
  h2 { color: #00e5ff; font-size: 1.4em; }
  h3 { color: #b39ddb; }
  code { background: #1e2030; color: #82aaff; border-radius: 4px; padding: 2px 6px; }
  pre { background: #1e2030; border-left: 4px solid #7c4dff; border-radius: 8px; }
  strong { color: #00e5ff; }
  em { color: #ce93d8; }
  a { color: #80cbc4; }
  blockquote { border-left: 4px solid #7c4dff; color: #b0bec5; background: #1a1d2e; padding: 12px 20px; border-radius: 0 8px 8px 0; }
  ul li::marker { color: #7c4dff; }
  section.title {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
  }
  section.title h1 { border: none; font-size: 2.8em; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }
  .highlight { background: #1e2030; border-radius: 8px; padding: 16px; border: 1px solid #7c4dff33; }
---

<!-- _class: title -->

# Large Language Models
## от токенов до генерации

**Для студентов, знакомых с PyTorch и MLP**

*"Всё, что нужно — это внимание"*

---

# Карта путешествия

```
Текст  →  Токены  →  Эмбеддинги  →  Трансформер  →  Логиты  →  Токены  →  Текст
  │           │            │               │               │
токени-   BPE/       позиционное      Attention       Softmax +
затор    WordPiece   кодирование     + MLP           Sampling
```

Мы пройдём каждый шаг от сырого текста до генерации.

> **Предпосылки:** вы умеете писать MLP в PyTorch, знаете backprop, понимаете `nn.Linear`, `nn.Embedding`, `CrossEntropyLoss`.

---

# Проблема: текст — не тензор

**Нейросеть хочет числа.** Текст — это символы.

Наивные варианты:

| Подход | Проблема |
|--------|----------|
| Один символ = один токен | Огромные последовательности, нет семантики |
| Одно слово = один токен | Гигантский словарь (>500k слов), OOV-проблема |
| Фиксированный n-gram | Комбинаторный взрыв |

**Решение:** *подслова (subwords)* — компромисс между символами и словами.

Пример:
```
"нейросеть" → ["ней", "ро", "сеть"]
"unbelievable" → ["un", "believ", "able"]
"ChatGPT" → ["Chat", "G", "PT"]
```

---

# Алгоритм BPE — Byte Pair Encoding

## Обучение токенизатора

**Инициализация:** словарь = все уникальные символы.

**Шаг слияния (повторять N раз):**
1. Найти наиболее частую **пару** соседних токенов в корпусе
2. Объединить её в один новый токен
3. Добавить в словарь

```
Корпус: "аа аб аа аб аа"

Итерация 1: пара "а","а" → "аа"   → словарь: {а, б, аа}
Итерация 2: пара "аа","б" → "ааб" → словарь: {а, б, аа, ааб}
```

![BPE визуализация](https://miro.medium.com/v2/resize:fit:1400/1*Y7g_HRatf5z2qnMZqPfDMg.png)

> GPT-4 использует **~100k** токенов. Русский текст токенизируется ~в 2-3 раза хуже (больше токенов на слово), чем английский.

---

# Кодировщик и декодировщик

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# Обучение
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=30000, min_frequency=2)
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Использование
encoded = tokenizer.encode("Large language models are powerful")
print(encoded.ids)     # [4821, 3012, 892, 347, 2891]
print(encoded.tokens)  # ['Large', ' language', ' models', ' are', ' power', 'ful']
```

**Важно:** токенизатор **не обучается вместе с моделью** — он обучается отдельно на корпусе текста и затем фиксируется.

---

# Эмбеддинги: токены → векторы

Словарь размером `V` → каждый токен получает вектор размерности `d_model`.

```python
# Это просто таблица поиска!
embedding = nn.Embedding(vocab_size=50257, embedding_dim=768)

token_ids = torch.tensor([4821, 3012, 892])   # [seq_len]
x = embedding(token_ids)                       # [seq_len, 768]
```

**Ключевое свойство:** похожие по смыслу токены оказываются близко в пространстве.

![Word2Vec Analogy](https://miro.medium.com/v2/resize:fit:828/format:webp/1*jpnudcxnQAzwPBtNs5MNFA.png)

```
king - man + woman ≈ queen
Москва - Россия + Франция ≈ Париж
```

> Эмбеддинги **обучаются** в процессе обучения модели через backprop.

---

# Проблема позиционности

MLP обрабатывает токены **независимо** — ей всё равно, в каком порядке стоят слова.

```
"Кошка съела мышь" = "Мышь съела кошку" ?  ← для простого MLP: ДА
```

Нам нужно закодировать **позицию** токена в последовательности.

## Sinusoidal Positional Encoding

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

![Positional Encoding Heatmap](https://machinelearningmastery.com/wp-content/uploads/2022/01/PE3.png)

```python
x = token_embeddings + positional_encoding  # просто сложение!
```

---

# Positional Encoding: интуиция

Каждая позиция — уникальный «отпечаток» в пространстве эмбеддингов.

**Почему синусы/косинусы?**
- Разные частоты → разные «масштабы» позиций
- Можно интерполировать на длины, которых не видели при обучении
- `PE(pos+k)` выражается линейно через `PE(pos)` → модель может учить относительные расстояния

**Современная альтернатива: RoPE (Rotary Positional Embedding)**

Вместо сложения — **вращение** вектора на угол, зависящий от позиции.
Используется в LLaMA, Mistral, GPT-NeoX.

> GPT-2 использовал обучаемые позиционные эмбеддинги (`nn.Embedding(max_len, d_model)`).

---

# Архитектура Трансформера

![Transformer Architecture](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

<div class="columns">

<div class="highlight">

**Encoder** (BERT-подобные)
- Видит весь контекст
- Двунаправленное внимание
- Задачи: классификация, NER, вопрос-ответ

</div>

<div class="highlight">

**Decoder** (GPT-подобные)
- Видит только прошлое
- Авторегрессионная генерация
- Задачи: генерация текста, LLM

</div>

</div>

Мы сосредоточимся на **decoder-only** (GPT-style) архитектуре.

---

# Блок Трансформера

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff   = FeedForward(d_model, d_ff)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-norm + residual connection
        x = x + self.attn(self.ln1(x))   # ← Self-Attention
        x = x + self.ff(self.ln2(x))     # ← MLP (FFN)
        return x
```

**Стек из N таких блоков** — это и есть трансформер.

GPT-3: N=96, d_model=12288, n_heads=96, d_ff=49152

---

# Внимание как Key-Value база данных

## Нечёткий поиск в памяти

Представьте словарь (БД) с мягкими ключами:

```
Обычный dict:   query → точное совпадение ключа → значение
Attention:      query → взвешенная сумма всех значений
```

| Аналогия | Attention |
|----------|-----------|
| Запрос к БД | **Query** `Q` |
| Ключи записей | **Keys** `K` |
| Значения записей | **Values** `V` |
| Результат | взвешенная сумма `V` |

> "Насколько каждый токен *релевантен* моему запросу?"

---

# Механизм Self-Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

```python
def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    # [B, heads, seq, seq] — матрица схожести
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)   # "внимание"
    return torch.matmul(weights, V)       # взвешенная сумма
```

![Attention Weights Visualization](https://lilianweng.github.io/posts/2018-06-24-attention/attention.png)

---

# Self-Attention: пошагово

Входной токен → **три** линейных проекции:

```python
Q = x @ W_Q   # что я ищу?
K = x @ W_K   # что я предлагаю?
V = x @ W_V   # что я передаю, если меня выбрали?
```

**Пример для фразы "банк реки":**

```
"реки" формирует Q: "ищу: водный объект"
"банк" формирует K: "я: финансовый институт"  → низкое сходство
       формирует K: "я: берег"                → высокое сходство
```

Модель **сама учится**, какие пары Q-K важны!

> Ключевое отличие от RNN: все токены обрабатываются **параллельно**, нет проблемы забывания.

---

# Causal (Masked) Self-Attention

Для **генерации** токен может видеть только **прошлое** (авторегрессия):

```python
# Маска: верхний треугольник = False (будущее закрыто)
mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

# На практике:
scores = scores.masked_fill(~mask, float('-inf'))
# После softmax: -inf → 0, будущее не влияет на настоящее
```

```
Токен 1 видит:  [1]
Токен 2 видит:  [1, 2]
Токен 3 видит:  [1, 2, 3]
    ...
```

![Causal Attention Mask](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_1_1080p.mov)

Это позволяет обучать **все позиции параллельно** на одном примере!

---

# Multi-Head Attention

**Идея:** запускаем несколько "голов" внимания параллельно — каждая смотрит на разные аспекты.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.heads, self.d_k).transpose(1,2)
        K = self.W_k(x).view(B, T, self.heads, self.d_k).transpose(1,2)
        V = self.W_v(x).view(B, T, self.heads, self.d_k).transpose(1,2)
        out = attention(Q, K, V)                            # [B, heads, T, d_k]
        out = out.transpose(1,2).contiguous().view(B, T, C) # concat heads
        return self.W_o(out)
```

---

# Что учат разные головы?

Разные головы специализируются на разных отношениях:

![Multi-Head Attention Patterns](https://miro.medium.com/v2/resize:fit:1400/1*uRPOkhX5UrG8V44qqJaL4w.png)

- **Голова 1:** синтаксические связи (подлежащее → сказуемое)
- **Голова 2:** кореференция (местоимения → существительные)
- **Голова 3:** позиционные паттерны (следующий/предыдущий токен)
- **Голова N:** семантические роли

> Это обнаруживается **эмпирически** — модель сама организует специализацию.

---

# Feed-Forward Network в трансформере

После attention — обычный **двуслойный MLP** применяется к каждому токену независимо:

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),   # d_ff = 4 * d_model обычно
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)  # применяется к каждому токену отдельно
```

**Интуиция:**
- Attention: **коммуникация** между токенами
- FFN: **обработка** информации внутри каждого токена

> FFN содержит ~2/3 всех параметров модели. Считается "памятью" модели.

---

# Полная GPT-подобная модель

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks  = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, 4*d_model)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)                            # [B, T, d]
        pos = self.pos_emb(torch.arange(T))               # [T, d]
        x = self.blocks(tok + pos)                        # [B, T, d]
        return self.head(self.ln_f(x))                    # [B, T, V]
```

---

# Обучение LLM: Language Modeling

## Задача: предсказать следующий токен

```python
# Одна строка текста — миллионы примеров обучения!
inputs:  "The quick brown fox"
targets: "quick brown fox jumps"
```

```python
def training_step(model, batch):
    x, y = batch          # x, y: [B, T]  (y = x сдвинутый на 1)
    logits = model(x)     # [B, T, vocab_size]
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),  # [B*T, V]
        y.view(-1)                    # [B*T]
    )
    return loss
```

**Это и есть всё обучение базовой LLM!**
Никаких специальных меток — текст сам является разметкой.

---

# Масштаб обучения LLM

| Модель | Параметры | Токены обучения | GPU-часы |
|--------|-----------|-----------------|----------|
| GPT-2 | 1.5B | 40B | ~100 A100 |
| GPT-3 | 175B | 300B | ~3500 A100·лет |
| LLaMA 3 | 70B | 15T | ~огромно |
| GPT-4 | ~1T? | ~13T? | неизвестно |

**Закон Чинчиллы** (Hoffmann et al., 2022):
> Оптимально: токены обучения ≈ 20× параметров

**AdamW optimizer** + **gradient checkpointing** + **mixed precision** (bf16) + **distributed training** (FSDP/DeepSpeed)

---

# Закон масштабирования (Scaling Laws)

![Scaling Laws](https://miro.medium.com/v2/resize:fit:1400/1*AFOQ3BO_mfJBVODaSw1VxQ.png)

Loss на следующий токен падает как **степенной закон** от:
- Числа параметров N
- Объёма данных D
- Количества вычислений C

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}$$

> **Вывод:** большие модели + больше данных = предсказуемое улучшение

---

# От базовой модели к ассистенту: RLHF

Базовая LLM умеет **продолжать текст**, но не следовать инструкциям.

**Reinforcement Learning from Human Feedback:**

```
1. Supervised Fine-Tuning (SFT)
   Базовая модель → дообучение на парах (инструкция, ответ)

2. Reward Model Training
   Люди ранжируют ответы модели → обучаем reward model

3. PPO (RL)
   Модель генерирует → reward model оценивает → 
   оптимизируем через PPO
```

![RLHF Pipeline](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/rlhf.png)

**Современная альтернатива: DPO** — обходится без reward model.

---

# Генерация: авторегрессия

```python
def generate(model, prompt_ids, max_new_tokens=100):
    x = prompt_ids.unsqueeze(0)  # [1, T]
    for _ in range(max_new_tokens):
        logits = model(x)[:, -1, :]   # [1, V] — только последний токен
        probs  = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1) # сэмплируем
        x = torch.cat([x, next_token], dim=1)    # добавляем к контексту
    return x
```

**Каждый новый токен** требует полного прохода модели.
Это делает генерацию медленной — отсюда **KV-cache**.

---

# KV-Cache: ускорение генерации

При авторегрессии K и V прошлых токенов **не меняются**!

```python
# БЕЗ кэша: O(T²) операций для T токенов
for step in range(T):
    Q, K, V = compute_qkv(all_tokens[:step])  # заново всё!
    output = attention(Q, K, V)

# С кэшем: O(T) операций
kv_cache = {}
for step in range(T):
    q, k, v = compute_qkv(new_token_only)
    kv_cache['k'] = torch.cat([kv_cache.get('k', k), k], dim=1)
    kv_cache['v'] = torch.cat([kv_cache.get('v', v), v], dim=1)
    output = attention(q, kv_cache['k'], kv_cache['v'])
```

> KV-cache — главная причина, почему **длинный контекст требует много памяти**.
> 128k токенов × 96 слоёв × 2 (K,V) × большие модели = десятки GB

---

# Стратегии сэмплирования

```python
logits = model(x)[:, -1, :]  # сырые логиты

# 1. Жадный поиск (Greedy)
next = logits.argmax(-1)  # всегда самый вероятный — скучно, повторяется

# 2. Temperature scaling
probs = F.softmax(logits / temperature, dim=-1)
# temperature < 1: острее (более детерминированно)
# temperature > 1: более равномерно (больше случайности)

# 3. Top-k sampling
top_k = 50
v, _ = logits.topk(top_k)
logits[logits < v[:, -1:]] = -float('Inf')
probs = F.softmax(logits, dim=-1)

# 4. Top-p (nucleus) sampling
sorted_probs, sorted_idx = torch.sort(probs, descending=True)
cumsum = sorted_probs.cumsum(dim=-1)
# убираем токены, где кумулятивная вероятность > p (например, 0.9)
```

---

# Сэмплирование: визуализация

![Sampling Strategies](https://miro.medium.com/v2/resize:fit:1400/1*bcVVKwkFVHMSNxMCo7uiuQ.png)

| Метод | Температура | Top-k | Top-p | Применение |
|-------|-------------|-------|-------|-----------|
| Greedy | — | — | — | Код (точность) |
| Beam Search | — | — | — | Перевод |
| Temperature | 0.7-0.9 | — | — | Общение |
| Top-p | 0.8 | — | 0.9 | Творческий текст |

> Хорошее значение по умолчанию: **temperature=0.8, top_p=0.9**

---

# Repetition Penalty

**Проблема:** модель может зацикливаться:
```
"Я думаю думаю думаю думаю думаю думаю..."
```

```python
def apply_repetition_penalty(logits, input_ids, penalty=1.2):
    for token_id in set(input_ids.tolist()):
        if logits[token_id] < 0:
            logits[token_id] *= penalty
        else:
            logits[token_id] /= penalty
    return logits
```

**Дополнительные трюки:**
- **Min-p sampling** — более новый и часто лучший метод
- **Typical sampling** — выбираем токены с типичной информационной ценностью
- **Beam search** — поддерживаем k лучших гипотез (но медленнее)

---

# Flash Attention: эффективность

Стандартный attention: **O(T²)** по памяти и времени.

**FlashAttention** (Dao et al., 2022) — переписывает attention под иерархию памяти GPU:

![Flash Attention](https://gordicaleksa.github.io/images/flash-attention/FA-fig1.png)

```python
# Вместо:
scores = Q @ K.T / sqrt(d)    # материализуем [T×T] матрицу
weights = softmax(scores)
out = weights @ V

# FlashAttention считает блоками — никогда не хранит полную T×T матрицу
from flash_attn import flash_attn_func
out = flash_attn_func(Q, K, V, causal=True)
```

> FlashAttention-2 даёт **2-4× ускорение** и позволяет обучать с длинными контекстами.

---

# Evaluation: как меряем качество LLM

## Perplexity — базовая метрика

$$PPL = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log P(w_t | w_{<t})\right)$$

```python
with torch.no_grad():
    logits = model(input_ids)          # [B, T, V]
    loss   = F.cross_entropy(
        logits[:, :-1].reshape(-1, V),
        input_ids[:, 1:].reshape(-1)
    )
    perplexity = torch.exp(loss)       # чем ниже — тем лучше
```

**Интуиция:** если PPL=10, модель в среднем "не могла решить" между 10 равновероятными токенами.

**Бенчмарки:** MMLU, HellaSwag, HumanEval, MATH, BIG-Bench...

---

# Соберём всё вместе

```
"What is 2+2?" 
    ↓  [Tokenizer]
[2061, 318, 362, 10, 19, 30]
    ↓  [Embedding + Positional Encoding]
Матрица эмбеддингов [6 × 768]
    ↓  [Transformer Block × 12]
    ├── MultiHead Attention (Q·Kᵀ/√d → softmax → ·V)
    └── FeedForward (Linear → GELU → Linear)
Контекстуальные представления [6 × 768]
    ↓  [LM Head: Linear(768, 50257)]
Логиты [6 × 50257]
    ↓  [Softmax + Sampling]
Следующий токен → "4"
    ↓  [Повторяем...]
"4" → decode → ответ
```

---

# Куда дальше?

<div class="columns">

<div class="highlight">

**Обязательно прочитать**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2 paper](https://openai.com/research/language-unsupervised)
- [Chinchilla paper](https://arxiv.org/abs/2203.15556)
- Karpathy: [nanoGPT](https://github.com/karpathy/nanoGPT)

</div>

<div class="highlight">

**Практика**
- Реализовать GPT с нуля (nanoGPT как гид)
- Поиграть с [LM Playground](https://lm-playground.org)
- Изучить [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Посмотреть [3Blue1Brown: Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc)

</div>

</div>

---

<!-- _class: title -->

# Спасибо!

## Итоги

Текст → **токенизация** (BPE) → **эмбеддинги** + **pos. encoding**
→ **N × [Attention + FFN]** → логиты → **сэмплирование** → текст

**Главная идея:**
> Self-Attention — это дифференцируемый нечёткий поиск в памяти,  
> где сеть сама учится формировать запросы, ключи и значения.

*Всё, что нужно — это внимание.*
