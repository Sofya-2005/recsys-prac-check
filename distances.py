import numpy as np

def jaccard_sim(ratings: np.array, user_vector: np.array) -> np.array:
    """
    Вычисляет похожести Жаккара между пользователем (user_vector)
    и всеми пользователями из матрицы оценок (ratings).
    
    :param ratings: матрица оценок размером (# пользователей, # предметов)
    :param user_vector: вектор оценок пользователя, для которого вычисляем схожесть
    :return: вектор значений похожести Жаккара (длиной ratings.shape[0])
    """

    jaccard_sim_arr = np.zeros(ratings.shape[0])
    

    user_set = user_vector != 0

    for i, user_ratings in enumerate(ratings):

        other_user_set = user_ratings != 0
        
        intersection = np.logical_and(user_set, other_user_set).sum()  
        union = np.logical_or(user_set, other_user_set).sum()          
        
        jaccard_sim_arr[i] = intersection / union if union != 0 else 0.0
    
    return jaccard_sim_arr
