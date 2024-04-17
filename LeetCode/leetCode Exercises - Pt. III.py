'79. Word Search'

# Input

# # Case 1
# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = 'ABCCED'
# # Output: True


# # Case 2
# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = 'SEE'
# # Output: True


# # Case 3
# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = 'ABCB'
# # Output: False


'''
Intuition:

    The problem can be solved by traversing the grid and performing a depth-first search (DFS) for each possible starting position. 
    At each cell, we check if the current character matches the corresponding character of the word. 
    If it does, we explore all four directions (up, down, left, right) recursively until we find the complete word or exhaust all possibilities.

    Approach

        1. Implement a recursive function backtrack that takes the current position (i, j) in the grid and the current index k of the word.
        2. Base cases:
            - If k equals the length of the word, return True, indicating that the word has been found.
            - If the current position (i, j) is out of the grid boundaries or the character at (i, j) does not match the character at index k of the word, return False.
        3. Mark the current cell as visited by changing its value or marking it as empty.
        4. Recursively explore all four directions (up, down, left, right) by calling backtrack with updated positions (i+1, j), (i-1, j), (i, j+1), and (i, j-1).
        5. If any recursive call returns True, indicating that the word has been found, return True.
        6. If none of the recursive calls returns True, reset the current cell to its original value and return False.
        7. Iterate through all cells in the grid and call the backtrack function for each cell. If any call returns True, return True, indicating that the word exists in the grid. Otherwise, return False.
        
'''


# # Backtracking (Recursive) Approach
# def exist(board: list[list[str]], word: str) -> bool:


#     def backtrack(i, j, k):

#         if k == len(word):
#             return True
        
#         if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[k]:
#             return False
        
#         temp = board[i][j]
#         board[i][j] = ''
        
#         if backtrack(i+1, j, k+1) or backtrack(i-1, j, k+1) or backtrack(i, j+1, k+1) or backtrack(i, j-1, k+1):
#             return True
        
#         board[i][j] = temp

#         return False
        


#     for i in range(len(board)):

#         for j in range(len(board[0])):

#             if backtrack(i, j, 0):

#                 return True
            

#     return False
        

# print(exist(board, word))




'xxx'