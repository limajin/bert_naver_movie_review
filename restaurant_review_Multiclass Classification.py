import tensorflow as tf
import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random
import time
import datetime

# 식당리뷰 1점 - 5점 까지 다중분류
data = pd.read_csv('min_score_count_data.csv')
# del data['Unnamed: 0']


# data['score'] = data['score'].astype(int)
# data['preprocessed_review'] = data['preprocessed_review'].astype(str)
data['score'] = data['score'].map({1: 0, 2: 1, 3: 2, 4: 3, 5: 4})
# data = data[0:100000].reset_index(drop=True)
data = data.dropna(axis=0)
data = data.reset_index(drop=True)
data.head()


# 훈련셋 테스트셋으로 분리
train_set, test_set = train_test_split(data[['preprocessed_review', 'score']],stratify = data.score, test_size=0.2, random_state=42)
print("train:", len(train_set), " test:", len(test_set))
print(train_set.score.mean())
print(test_set.score.mean())

# 리뷰 문장 추출
sentences = train_set['preprocessed_review']
# sentences[:5]

# BERT의 입력 형식에 맞게 변환
sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]

# 점수 추출
scores = train_set['score'].values
print("train_set_scores:", scores)

# BERT의 토크나이저로 문장을 토큰으로 분리
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

# print (sentences[0])
# print (tokenized_texts[0])


# 입력 토큰의 최대 시퀀스 길이
MAX_LEN = 200

# 토큰을 숫자 인덱스로 변환
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# 어텐션 마스크 초기화
attention_masks = []

# 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
# 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

# print(attention_masks[0])

# 훈련셋과 검증셋으로 분리
X_train, X_test, y_train, y_test = train_test_split(input_ids, scores, stratify=scores,random_state=42, test_size=0.2)
print(y_train.mean())
print(y_test.mean())
# 어텐션 마스크를 훈련셋과 검증셋으로 분리
train_masks, test_masks, _, _ = train_test_split(attention_masks,
                                                 input_ids,
                                                 random_state=42,
                                                 test_size=0.2)
# 데이터를 파이토치의 텐서로 변환
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
train_masks = torch.tensor(train_masks)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)
test_masks = torch.tensor(test_masks)

# print(X_train[0])
# print(y_train[0])
# print(train_masks[0])
# print(X_test[0])
# print(y_test[0])
# print(test_masks[0])
#

# 배치 사이즈
batch_size = 24

# 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
# 학습시 배치 사이즈 만큼 데이터를 가져옴
train_data = TensorDataset(X_train, train_masks, y_train)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(X_test, test_masks, y_test)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# 전처리 - 테스트셋
# 리뷰 문장 추출
sentences = test_set['preprocessed_review']

# BERT의 입력 형식에 맞게 변환
sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]

# 점수 추출
scores = test_set['score'].values
print("test_set_score:", scores)

# BERT의 토크나이저로 문장을 토큰으로 분리
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
#
# print (sentences[0])
# print (tokenized_texts[0])

# 입력 토큰의 최대 시퀀스 길이
MAX_LEN = 200

# 토큰을 숫자 인덱스로 변환
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# 어텐션 마스크 초기화
attention_masks = []

# 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
# 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

# 데이터를 파이토치의 텐서로 변환
X_test = torch.tensor(input_ids)
y_test = torch.tensor(scores)
test_masks = torch.tensor(attention_masks)

# 배치 사이즈
batch_size = 24

# 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
# 학습시 배치 사이즈 만큼 데이터를 가져옴
test_data = TensorDataset(X_test, test_masks, y_test)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# 디바이스 설정
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

# 분류를 위한 BERT 모델 생성
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=5)
model.cuda()

# 옵티마이저 설정
optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # 학습률
                  eps=1e-8  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                  )

# 에폭수
epochs = 10

# 총 훈련 스텝 : 배치반복 횟수 * 에폭
total_steps = len(train_dataloader) * epochs

# 처음에 학습률을 조금씩 변화시키는 스케줄러 생성
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)


# 정확도 계산 함수
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# 시간 표시 함수
def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))

    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))


# 재현을 위해 랜덤시드 고정
seed_val = 32
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 그래디언트 초기화
model.zero_grad()

# 에폭만큼 반복
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # 시작 시간 설정
    t0 = time.time()

    # 로스 초기화
    total_loss = 0

    # 훈련모드로 변경
    model.train()

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for step, batch in enumerate(train_dataloader):
        # 경과 정보 표시
        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)


        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = torch.tensor(b_input_ids).to(device).long()
        # print(b_labels)

        # Forward 수행
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        # 로스 구함
        loss = outputs[0]

        # 총 로스 계산
        total_loss += loss.item()

        # Backward 수행으로 그래디언트 계산
        loss.backward()

        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 그래디언트를 통해 가중치 파라미터 업데이트
        optimizer.step()

        # 스케줄러로 학습률 감소
        scheduler.step()

        # 그래디언트 초기화
        model.zero_grad()

    # 평균 로스 계산
    avg_train_loss = total_loss / len(train_dataloader)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    # 시작 시간 설정
    t0 = time.time()

    # 평가모드로 변경
    model.eval()

    # 변수 초기화
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for batch in test_dataloader:
        # 배치를 device를 넣음
        batch = tuple(t.to(device) for t in batch)
        # print(batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = torch.tensor(b_input_ids).to(device).long()
        # print(b_labels)

        # 그래디언트 계산 안함
        with torch.no_grad():
            # Forward 수행
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # 로스 구함
        logits = outputs[0]

        # CPU로 데이터 이동
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # 출력 로짓과 라벨을 비교하여 정확도 계산
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

torch.save(model, 'results/bert_model.pt' + datetime.datetime.now().strftime('%Y-%m-%d_%H_%M'))  # 전체 모델 저장
torch.save(model.state_dict(), 'results/bert_model_state_dict.pt' + datetime.datetime.now().strftime('%Y-%m-%d_%H_%M'))

print("")
print("Training complete!")

print("\n")
print("========== Testing ==========")
# 시작 시간 설정
t0 = time.time()

# 평가모드로 변경
model.eval()

# 변수 초기화
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

# 데이터로더에서 배치만큼 반복하여 가져옴
for step, batch in enumerate(test_dataloader):
    # 경과 정보 표시
    if step % 100 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

    # 배치를 GPU에 넣음
    batch = tuple(t.to(device) for t in batch)


    # 배치에서 데이터 추출
    b_input_ids, b_input_mask, b_labels = batch
    b_input_ids = torch.tensor(b_input_ids).to(device).long()

    # 그래디언트 계산 안함
    with torch.no_grad():
        # Forward 수행
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

    # 로스 구함
    logits = outputs[0]

    # 총 로스 계산
    total_loss += loss.item()

    # 평균 로스 계산
    avg_test_loss = total_loss / len(test_dataloader)

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # 출력 로짓과 라벨을 비교하여 정확도 계산
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

print("")
print("Average test loss: {0:.2f}".format(avg_test_loss))
print("Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
print("Test took: {:}".format(format_time(time.time() - t0)))