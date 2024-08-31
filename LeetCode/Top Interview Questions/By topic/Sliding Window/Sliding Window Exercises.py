'''
CHALLENGES INDEX

3. Longest Substring Without Repeating Characters (Hash Table) (SW)


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
*Arrays, Hash Tables & Matrices
*Sorting
*Heaps, Stacks & Queues
*Graphs, Trees & Binary Trees

(XX)
'''


'3. Longest Substring Without Repeating Characters'
# def x():

#     s = "abcabcbb"


#     # My solution
#     substrings = []

#     i = 0

#     while i < len(s):

#         sub = str()

#         for char in s[i:]:

#             if char in sub:
#                 substrings.append(sub)
#                 break

#             sub += char
        
#         if sub not in substrings:
#             substrings.append(sub)

#         i += 1

#     # print(substrings)

#     max_sub = max(substrings, key = len) if substrings else 0

#     # print(max_sub)

#     print(max_sub, len(max_sub))


#     # Another more efficient solution

#     def lengthOfLongestSubstring(s: str) -> int:
            
#             n = len(s)
#             maxLength = 0
#             charMap = {}
#             left = 0
            
#             for right in range(n):

#                 if s[right] not in charMap or charMap[s[right]] < left:
#                     charMap[s[right]] = right
#                     maxLength = max(maxLength, right - left + 1)

#                 else:
#                     left = charMap[s[right]] + 1
#                     charMap[s[right]] = right
            
#             return maxLength


#     lengthOfLongestSubstring(s)











