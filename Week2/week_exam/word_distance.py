def levenshtein_distance(source, target):
    matrix = [[0 for _ in range(len(target) + 1)] for _ in range(len(source) + 1)]

    for i in range(len(source)+1):
        matrix[i][0] = i
    
    for j in range(len(target)+1):
        matrix[0][j] = j  
      
    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            if source[i-1] == target[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                insert_cost = matrix[i][j-1] + 1
                delete_cost = matrix[i-1][j] + 1
                sub_cost = matrix[i-1][j-1] + 1  #Chi phí thay thế nếu 2 ký tự khác nhau
                matrix[i][j] = min(insert_cost, delete_cost, sub_cost)
                
    return matrix[i][j]   
           
def main():
    source = "hola"
    target = "hello"
    assert levenshtein_distance("hi", "hello") == 4
    print(f"cần {levenshtein_distance(source, target)} lượt để thay đổi từ {source} sang {target}")

if __name__ == "__main__":
    main()