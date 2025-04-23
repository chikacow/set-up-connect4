#Có 1 vấn đề, đó chính là những nước đi của mình ở đây và trên server có giống nhau ko, giống thì ok, ko giống thì đứt, đứt luôn

#Có thể cả tiến hàm find new move để tìm riêng new move của 1 và 2!!! to solve the problem above

#phía dưới là vòng for giả lập 5 turn đầu tiên

def find_new_move_v1(prev_board, curr_board):
    for col in range(7):
        for row in range(6):
            if prev_board[row][col] == 0 and curr_board[row][col] in (1, 2):
                return col  # Trả về index cột mới có nước đi 0-6
    return -1  # Không tìm thấy khác biệt

def find_new_move_v2(board_before, board_after):
    """
    Hàm nhận vào hai bảng Connect 4: trước và sau lượt chơi.
    Trả về tuple gồm cột mà người chơi 1 và người chơi 2 vừa đi.
    Nếu không có nước đi mới thì trả về None.
    """
    move_p1 = None
    move_p2 = None

    for row in range(6):
        for col in range(7):
            if board_before[row][col] != board_after[row][col]:
                if board_after[row][col] == 1:
                    move_p1 = col
                elif board_after[row][col] == 2:
                    move_p2 = col

    return move_p1, move_p2


def drop_piece(board, col_index, player):
    """
    Thêm quân vào cột được chọn (col_index), theo đúng logic rơi từ trên xuống.
    player = 1 hoặc 2
    """
    for row in reversed(range(6)):  # bắt đầu từ hàng dưới lên
        if board[row][col_index] == 0:
            board[row][col_index] = player
            return True  
    return False  # cột đã đầy, ko thể thả

def print_board(board):
    for row in board:
        print(" ".join(map(str, row))) 
    
#kiểm tra xem mảng có phải mảng khởi tạo khi bắt đầu game ko
def is_full_of_zeros(board):
    for row in board:
        if any(cell != 0 for cell in row):  # Kiểm tra nếu có phần tử nào khác 0 trong hàng
            return False
    return True  # Nếu tất cả các phần tử là 0


#global
move_history = "" 

import random
#AI solution
def make_move(str):
    return random.randint(0, 6)

def analysv1(prev, curr):
    global move_history
    if is_full_of_zeros(curr):
        my_move = make_move(move_history)
        move_history += str(my_move)
        
        drop_piece(curr, my_move, 1) #player = 1
        #prev = [row[:] for row in curr]
        
        #clone prev
        drop_piece(prev, my_move, 1)
        
        
        print("from begining")
        
    
    else:
        opp_move = find_new_move_v1(prev, curr)
        
        print("opponent plays move at: " + str(opp_move))
        
        move_history += str(opp_move)
        
        my_move = make_move(move_history)
        
        move_history += str(my_move)
        
        
        
        drop_piece(curr, my_move, 1) #player = 1
        
        
        #prev = [row[:] for row in curr]
        
        #clone prev
        drop_piece(prev, opp_move, 2) #order is important
        drop_piece(prev, my_move, 1)
    
    
    return move_history
    

def analysv2(prev, curr):
    global move_history
    
    #ban dau mang prev va curr deu = 0
    #curr is the board that we receive from server
    
    if is_full_of_zeros(curr):
        my_move = make_move(move_history)
        move_history += str(my_move)
        
        prev = curr
        
        
        print("from begining")
        
    else:
        player_move,opponent_move = find_new_move_v2(prev, curr)
        
        print("player plays move at: " + str(player_move))
        print("opponent plays move at: " + str(opponent_move))
        
        
        move_history += str(opponent_move)
        move_history += str(player_move)
        
        my_move = make_move(move_history)
        
        
        prev = curr
    
    
    return move_history
    
#global  
prev = [
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0]
] #initial

#global
curr = [
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0]
] #initial

#print(analys(prev, curr))

#cuz im lazy af
last_shit = ""

for i in range(1,6):
    print("initial/received state")
    print_board(curr)
    print("______________________")
    print(analysv1(prev, curr))
    print_board(prev)
    print("######################")
    
    
    #fake AI - pretending to be the server to create moves and return an updated board, which is curr for here
    simu_opp = make_move(str)
    drop_piece(curr, simu_opp, 2) #opp = 2
    
    
    
    last_shit = str(simu_opp) #hard af


    print_board(curr)

final_move_history = move_history + last_shit

print("final: " + final_move_history )

#output

# from begining
# 3
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 1 0 0 0
# ______________________
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 1 0 2 0
# opponent plays move at: 5
# 350
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 1 0 0 1 0 2 0
# ______________________
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 1 0 2 1 0 2 0
# opponent plays move at: 2
# 35020
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 1 0 0 0 0 0 0
# 1 0 2 1 0 2 0
# ______________________
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 1 0 0 2 0 0 0
# 1 0 2 1 0 2 0
# opponent plays move at: 3
# 3502035
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 1 0 0 2 0 1 0
# 1 0 2 1 0 2 0
# ______________________
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 2 0 0 0
# 1 0 0 2 0 1 0
# 1 0 2 1 0 2 0
# opponent plays move at: 3
# 350203532
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 2 0 0 0
# 1 0 1 2 0 1 0
# 1 0 2 1 0 2 0
# ______________________
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 2 0 0 0
# 0 0 0 2 0 0 0
# 1 0 1 2 0 1 0
# 1 0 2 1 0 2 0
# final: 3502035323
