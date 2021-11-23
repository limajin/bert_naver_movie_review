import torch

# transformers: BERT 이용 메인 API
# torch: 파인튜닝 실제 학습용 딥러닝 프레임워크
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

from tqdm import tqdm


# 네이버 영화 리뷰데이터 다운로드 터미널에 진행
# nsmc라는 이름의 디렉토리가 생성됨
# !git clone https://github.com/e9t/nsmc.git

# train, test 데이터 불러오기
train_data = pd.read_csv("nsmc/ratings_train.txt", sep='\t')
test_data = pd.read_csv("nsmc/ratings_test.txt", sep='\t')
print(train_data.shape) # (150000, 3)
print(test_data.shape) # (50000, 3)

# 전처리
# BERT 분류모델은 각 문장의 앞마다 [CLS]를 붙이고, 종료는 [SEP]
pre_bert = ["[CLS] " + str(s) + " [SEP]" for s in train_data.document]
print(pre_bert[:5])
# test 데이터도 동일하게
test_sentences = test_data['document']
test_sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in test_sentences]
labels = test_data['label'].values

# 토크나이징
# 사전학습된 BERT multilingual 모델 내 포함된 토크나이저를 활용
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

# 한 문장을 토크나이징하여 리스트에 담음
# list[0]에는 한 문장을 토크나이징한 결과가 담긴다
tokenized_texts = [tokenizer.tokenize(s) for s in pre_bert]
test_tokenized_texts = [tokenizer.tokenize(sent) for sent in test_sentences]

print(tokenized_texts)

# 패딩
# 전체 훈련 데이터에서 각 샘플의 길이는 서로 다를 수 있음
# 모델의 입력으로 사용하려면 모든 샘플의 길이를 동일하게 맞추어야할 때
# 보통 숫자 0을 넣어서 길이가 다른 샘플들의 길이를 맞춰줌 -> 패딩
# 케라스에서 pad_sequence()를 사용해 정해준 길이보다 길이가 긴 샘플은 값을 일부 자르고, 정해준 길이보다 길이가 짧은 샘플은 값을 0으로 채웁니다.
MAX_LEN = 128 #token의 max length보다 크게 설정

# convert_tokens_to_ids: 분할된 단어목록을 ID로 변환하는 함수
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# 설정한 MAX_LEN 만큼 빈 공간을 0이 채움
# : 가변 길이 시퀀스를 채우기 위해 사용, 디폴트 패딩 값은 0.0
# 포스트 시퀀스 패딩할 경우 padding='post' - 시퀀스의 끝 부분에 패딩 적용 (사전 시퀀스 패딩이 기본값임)
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
input_ids[0]

test_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in test_tokenized_texts]
test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


# 어텐션 마스크
# 학습속도를 높이기 위해 실 데이터가 있는 곳과 padding이 있는 곳을 attention에게 알려줌
attention_masks = []

# 실 데이터가 있는 곳은 1.0 padding은 0.0
for seq in tqdm(input_ids):
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

print(attention_masks[0])

test_attention_masks = []
for seq in tqdm(test_input_ids):
    seq_mask = [float(i>0) for i in seq]
    test_attention_masks.append(seq_mask)

# train - validation set 분리
# 마스크와 input이 달라지지 않도록 random_state는 일정하게 고정
train_inputs, validation_inputs, train_labels, validation_labels = \
train_test_split(input_ids, train_data['label'].values, random_state=42, test_size=0.1)

# test는 처음에 로드할 때 나눠놨기 때문에 train과 validation set만 분리
train_masks, validation_masks, _, _ = train_test_split(attention_masks,
                                                       input_ids,
                                                       random_state=42,
                                                       test_size=0.1)


# numpy ndarray를 torch tensor로 변환
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)

test_inputs = torch.tensor(test_input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(test_attention_masks)



# 배치 및 데이터로더 설정 - GPU 사용할 경우
# GPU의 VRAM에 맞도록 배치사이즈를 설정
# VRAM 부족 메시지 ->  8의 배수 중 더 작은 것으로 줄여나갈 것
BATCH_SIZE = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)


test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)



# 모델 학습 - gpu, cpu 택
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')


# BERT 모델 생성
# 이진분류 num_labels는 2
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.cuda()

# 옵티마이저
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # 학습률
                  eps = 1e-8 # 0으로 나누는 것을 방지하기 위한 epsilon 값
                )
# 에폭수
epochs = 4

# 총 훈련 스텝
total_steps = len(train_dataloader) * epochs

# lr 조금씩 감소시키는 스케줄러
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

# 정확도 계산 함수
def accuracy_cal(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# 시간 표시 함수
def watch_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))
    # hh:mm:ss 형태로
    return str(datetime.timedelta(seconds=elapsed_rounded))



# 학습 실행

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 그래디언트 초기화
model.zero_grad()

# 에폭만큼 반복
for epoch_i in range(0, epochs):


    #               Training

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
            elapsed = watch_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # 배치를 GPU/CPU 에 넣음
        batch = tuple(t.to(device) for t in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch

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
    print("  Training epcoh took: {:}".format(watch_time(time.time() - t0)))




    #               Validation

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
    for batch in validation_dataloader:
        # 배치를 GPU/CPU 에 넣음
        batch = tuple(t.to(device) for t in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch

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
        tmp_eval_accuracy = accuracy_cal(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(watch_time(time.time() - t0)))

print("")
print("Training complete!")