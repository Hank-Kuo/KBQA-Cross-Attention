import torch
import model.data_loader as data_loader

def evaluate(data, device, model, word2idx, entity2idx):
    model.eval()
    answers = []
    data_gen = data_loader.data_generator(data=data, word2ix=word2idx, entity2idx=entity2idx)
    total_correct = 0
    error_count = 0
    for i in range(len(data)):
        try:
            d = next(data_gen)
            head = d[0].to(device)
            question = d[1].to(device)
            ans = d[2]
            ques_len = d[3].unsqueeze(0)
            tail_test = torch.tensor(ans, dtype=torch.long).to(device)
            top_2 = model.get_score_ranked(head=head, sentence=question, sent_len=ques_len)
            top_2_idx = top_2[1].tolist()[0]
            head_idx = head.tolist()
            if top_2_idx[0] == head_idx:
                pred_ans = top_2_idx[1]
            else:
                pred_ans = top_2_idx[0]
            if type(ans) is int:
                ans = [ans]
            is_correct = 0
            if pred_ans in ans:
                total_correct += 1
                is_correct = 1
            q_text = d[-1]
            answers.append(q_text + '\t' + str(pred_ans) + '\t' + str(is_correct))
        except:
            error_count += 1
            
    print(error_count)
    accuracy = total_correct/len(data)
    return answers, accuracy