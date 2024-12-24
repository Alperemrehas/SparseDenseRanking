from sklearn.metrics import ndcg_score, precision_score, recall_score

def evaluate(predictions, ground_truths):
    ndcg = ndcg_score([ground_truths], [predictions])
    precision = precision_score(ground_truths, predictions, average='macro')
    recall = recall_score(ground_truths, predictions, average='macro')
    return {'NDCG': ndcg, 'Precision': precision, 'Recall': recall}
