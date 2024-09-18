'''
CHALLENGES TAGS

*LL: Linked-Lists
*BS: Binary Search
*DP: Dynamic Programming
*RC: Recursion
*TP: Two-pointers
*FCD: Floyd's cycle detection (Hare & Tortoise approach)
*PS: Preffix-sum
*SW: Sliding-Window
*MEM: Memoization
*GRE: Greedy
*DQ: Divide and Conquer
*BT: Backtracking
*BFS & DFS: Breadth-First Search & Depth-First Search
*Arrays, Hash Tables & Matrices
*Sorting
*Heaps, Stacks & Queues
*Graphs & Trees
*Others

'''

#Template
'xxx'
'''xxx'''
# def x():

#     from typing import Optional

#     # Base
#     # Definition for singly-linked list.
#     class ListNode:
#         def __init__(self, val=0, next=None):
#             self.val = val
#             self.next = next

#     # Input
#     # Case 1
#     head = ListNode(1, next=ListNode(2,next=ListNode(3,next=ListNode(4))))
#     # Output: [2,1,4,3]

#     '''
#     My Approach

#         Intuition:
            
#             -...
#     '''

#     def x() -> int:

#         # Handle Corner case: ...
#         if not head:
#             return
                
#         # Return ...
#         return 

#     # Testing
#     print(x())

#     '''Note: Done'''



'51. N-Queens (Array) (Matrix) (BT)'
'''51. N-Queens'''
def x():

    from typing import Optional

    # Input
    # Case 1
    n = 4
    # Output: [
    #   [".Q..","...Q","Q...","..Q."],
    #   ["..Q.","Q...","...Q",".Q.."]
    # ]

    # Case 2
    n = 1
    # Output: [Q]
    
    # Case 2
    n = 9
    # Output: [Q]

    '''
    Solution

        Explanation
            
            -...
    '''

    def solveNQueens(n):

        def is_valid(board, row, col):

            # Check the column
            for i in range(row):
                if board[i] == col:
                    return False
                
            # Check the diagonal (left-up)
            for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
                if board[i] == j:
                    return False
                
            # Check the diagonal (right-up)
            for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
                if board[i] == j:
                    return False
                
            return True


        def backtrack(row, board, solutions):

            if row == n:

                # If all queens are placed, convert the board to the required output format
                solutions.append(["." * col + "Q" + "." * (n - col - 1) for col in board])
                return
            
            for col in range(n):

                if is_valid(board, row, col):
                    
                    board[row] = col
                    backtrack(row + 1, board, solutions)
                    board[row] = -1  # Backtrack


        # Initialize board and solutions list
        solutions = []
        board = [-1] * n  # Keeps track of queen's position in each row
        backtrack(0, board, solutions)

        return solutions


    # Testing
    solution = solveNQueens(n=n)
    print(f'# of Solution: {len(solution)}')
    for i in solution[1]:
        print(i)

    '''Note: Done'''



x()




















































