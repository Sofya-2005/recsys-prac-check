from typing import List

def _compute_binary_relevance(
    recommended_items_list: List[int],
    true_items_list: List[int],
) -> List[int]:
  # your code here:
  return [1 if item in true_items_list else 0 for item in recommended_items_list]


def ap_at_k(
    recommended_items_list: List[int],
    true_items_list: List[int],
    k: int
) -> float:
  # your code here:
    recommended_items_list = recommended_items_list[:k]
    
    relevance = _compute_binary_relevance(recommended_items_list, true_items_list)
    
    precision_sum = 0
    num_relevant = 0
    
    for i, rel in enumerate(relevance, start=1):
        if rel == 1:  
            num_relevant += 1
            precision_sum += num_relevant / i 
            
    return precision_sum / min(len(true_items_list), k) if len(true_items_list) > 0 else 0.0



def map_at_k(
    recommended_items_lists: List[List[int]],
    true_items_lists: List[List[int]],
    k: int,
) -> float:
  """
  Computes ap@k for all buyers
  """
  assert len(recommended_items_lists) == len(true_items_lists), \
  'len(true_items_list) != len(recommended_items_list)'

  # your code here:
  assert len(recommended_items_lists) == len(true_items_lists), \
        'len(true_items_list) != len(recommended_items_list)'
    
  average_precisions = [
        ap_at_k(recommended_items_list, true_items_list, k)
        for recommended_items_list, true_items_list in zip(recommended_items_lists, true_items_lists)
    ]
    
  return sum(average_precisions) / len(average_precisions) if len(average_precisions) > 0 else 0.0