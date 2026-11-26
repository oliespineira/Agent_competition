"""Chess agent for the AI Agent Tournament (AI: Reasoning & Problem Solving, IE University).

Map of this file to the course:
  PART 1  Evaluation function E(s) = sum of weighted features --> Session 15
  PART 2  Required functions: evaluate, minimax, alphabeta --> Sessions 14 and 15
  PART 3  Playing search: alpha-beta with depth cutoff, move ordering,
          iterative deepening, transposition table, quiescence --> Session 14 + handbook extensions
  PART 4  act(): clock management and the decision --> Session 2 (agent), handbook

Convention used EVERYWHERE in this file: scores are from WHITE's point of view.
White is MAX (wants high scores), Black is MIN (wants low scores), exactly like the
MAX/MIN trees of Session 14. Scores are in centipawns: 100 = one pawn.
"""
import time                 # to measure how long we have been thinking

import chess                # board representation, legal moves and rules (allowed by the handbook)
import chess.polyglot       # only for zobrist_hash(): a fast, public way to turn a position into a number


# =====================================================================================
# PART 1. EVALUATION FUNCTION (Session 15: "E(s) = sum of w_i * f_i(s)")
# =====================================================================================

MATE = 100000               # score of a checkmate: much larger than any material difference
MATE_THRESHOLD = MATE - 1000  # any |score| above this means "a forced mate was found"
INFINITY = 10 ** 9          # bigger than every possible score; starting value for alpha and beta

# Feature 1: MATERIAL. Session 15 uses pawn 1, knight/bishop 3, rook 5, queen 9.
# We use the same idea in centipawns (bishop slightly above knight, a common refinement).
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,          # the king is never captured, so it has no material value
}

# Feature 2: POSITION (piece-square tables).
# These tables implement the "CONTROL" and "KING SAFETY" features from Session 15.
# A piece is not worth the same on every square: a knight in the centre attacks up to
# 8 squares, while in a corner it attacks only 2. So each piece type gets an 8x8 table
# that adds a bonus (or a penalty) depending on where the piece stands:
#   - CONTROL: knights, bishops and pawns score higher in the centre.
#   - KING SAFETY: in the middlegame the king scores higher when it has castled and
#     stays behind its pawns.
# Instead of counting attacked squares or checking the pawn shield on every evaluation,
# we just look up one number per piece, which is much cheaper.
# The values come from T. Michniewski's "Simplified Evaluation Function" (Chess
# Programming Wiki); we did not tune them ourselves.
# Tables are written from White's point of view, as you see the board from White's side:
# the first row is rank 8 and the last row is rank 1. For Black, table_bonus() flips the
# table vertically, so the same table works for both colours.
PAWN_TABLE = [
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [ 50,  50,  50,  50,  50,  50,  50,  50],   # about to promote: big bonus
    [ 10,  10,  20,  30,  30,  20,  10,  10],
    [  5,   5,  10,  25,  25,  10,   5,   5],
    [  0,   0,   0,  20,  20,   0,   0,   0],   # central pawns are good
    [  5,  -5, -10,   0,   0, -10,  -5,   5],
    [  5,  10,  10, -20, -20,  10,  10,   5],   # do not leave d2/e2 pawns at home
    [  0,   0,   0,   0,   0,   0,   0,   0],
]
KNIGHT_TABLE = [
    [-50, -40, -30, -30, -30, -30, -40, -50],   # knights on the edge are weak
    [-40, -20,   0,   0,   0,   0, -20, -40],
    [-30,   0,  10,  15,  15,  10,   0, -30],
    [-30,   5,  15,  20,  20,  15,   5, -30],   # knights in the centre are strong
    [-30,   0,  15,  20,  20,  15,   0, -30],
    [-30,   5,  10,  15,  15,  10,   5, -30],
    [-40, -20,   0,   5,   5,   0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50],
]
BISHOP_TABLE = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [-10,   5,   5,  10,  10,   5,   5, -10],
    [-10,   0,  10,  10,  10,  10,   0, -10],
    [-10,  10,  10,  10,  10,  10,  10, -10],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20],
]
ROOK_TABLE = [
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  5,  10,  10,  10,  10,  10,  10,   5],   # a rook on the 7th rank attacks pawns
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [  0,   0,   0,   5,   5,   0,   0,   0],
]
QUEEN_TABLE = [
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,   5,   5,   5,   0, -10],
    [ -5,   0,   5,   5,   5,   5,   0,  -5],
    [  0,   0,   5,   5,   5,   5,   0,  -5],
    [-10,   5,   5,   5,   5,   5,   0, -10],
    [-10,   0,   5,   0,   0,   0,   0, -10],
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
]
KING_MIDDLEGAME_TABLE = [                       # KING SAFETY: stay behind the pawns, castle
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [ 20,  20,   0,   0,   0,   0,  20,  20],
    [ 20,  30,  10,   0,   0,  10,  30,  20],   # castled king (g1 or c1) gets a bonus
]
KING_ENDGAME_TABLE = [                          # with few pieces left the king must be active
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10,   0,   0, -10, -20, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],   # centre is best ...
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -30,   0,   0,   0,   0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50],   # ... and the corner is worst (helps to checkmate a lone king)
]
TABLE_FOR = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_MIDDLEGAME_TABLE  # Default to middlegame table
}

# Feature weights w_i (Session 15: "these weights are modelling choices").
W_MATERIAL = 1.0
W_POSITION = 1.0


def table_bonus(table, square, color):
    """Read the bonus of a piece of `color` standing on `square` from a table written for White."""
    rank = chess.square_rank(square)    # 0 = rank 1 ... 7 = rank 8
    file = chess.square_file(square)    # 0 = file a ... 7 = file h
    if color == chess.WHITE:
        row = 7 - rank                  # the table's first row is rank 8, so rank 1 is row 7
    else:
        row = rank                      # Black sees the board upside down: its rank 1 is our rank 8 (row 0)
    return table[row][file]


def build_scores(king_table):
    """Precompute the value of every piece on every square, once, when the file loads.

    For each colour, piece type and square we store
        w1 * material + w2 * position bonus,
    positive for White and negative for Black, so a position's score is simply the sum
    of the values of the pieces on the board.
    The numbers are exactly the ones we would compute during the game. The difference is
    that the search only has to look them up instead of recalculating them at every node.
    That makes the evaluation about twice as fast, so the search can go deeper in the
    same time.
    `king_table` is the only table that changes between the middlegame and endgame
    versions."""
    scores = {}
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1               # White adds, Black subtracts
        for piece_type in PIECE_VALUE:
            table = king_table if piece_type == chess.KING else TABLE_FOR[piece_type]
            scores[color, piece_type] = [
                sign * (W_MATERIAL * PIECE_VALUE[piece_type] + W_POSITION * table_bonus(table, square, color))
                for square in chess.SQUARES                    # chess.SQUARES = 0 (a1) ... 63 (h8)
            ]
    return scores


MIDDLEGAME_SCORES = build_scores(KING_MIDDLEGAME_TABLE)   # used while queens and pieces remain
ENDGAME_SCORES = build_scores(KING_ENDGAME_TABLE)         # only the king's table differs


def is_endgame(board):
    """Endgame rule from the same source: no queens, or every side that has a queen has
    at most one minor piece (knight or bishop) and no rook besides it. Then the king should come out."""
    for color in (chess.WHITE, chess.BLACK):
        if board.pieces_mask(chess.QUEEN, color):                  # this side still has a queen
            minors = chess.popcount(board.pieces_mask(chess.KNIGHT, color) | board.pieces_mask(chess.BISHOP, color))
            rooks = chess.popcount(board.pieces_mask(chess.ROOK, color))
            if rooks > 0 or minors > 1:                            # queen plus real support: not an endgame
                return False
    return True


def static_evaluation(board):
    """Estimate how good an unfinished position is, E(s) from Session 15.

    E(s) = w1 * material + w2 * position. Since build_scores() already combined both
    terms for every piece on every square, here we just add up the precomputed value of
    each piece on the board. White's pieces add and Black's subtract.
    The score is always from White's point of view: positive means White is better,
    negative means Black is better.
    We first check whether the game is in the endgame, because the king should hide in
    the middlegame but become active in the endgame, and use the matching set of scores.
    Finished games (checkmate, draws) are handled by evaluate(), not here."""
    scores = ENDGAME_SCORES if is_endgame(board) else MIDDLEGAME_SCORES
    total = 0
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in PIECE_VALUE:                             # pawn, knight, ..., king
            row = scores[color, piece_type]                        # 64 precomputed scores for this piece
            # pieces_mask gives a 64-bit number with one bit per square holding such a piece;
            # scan_forward turns it into the list of those squares.
            for square in chess.scan_forward(board.pieces_mask(piece_type, color)):
                total += row[square]
    return int(total)


def game_is_over(board):
    """Terminal test under the tournament rules: python-chess already detects checkmate,
    stalemate and insufficient material. The fifty-move rule needs extra care: in normal
    chess a player has to claim the draw, so python-chess does not end the game by itself.
    In the tournament the draw is  automatic after fifty moves with no capture and no pawn 
    move (100 half-moves), so we check that counter ourselves."""
    return board.is_game_over() or board.halfmove_clock >= 100


# =====================================================================================
# PART 2. REQUIRED FUNCTIONS (the judge tests these separately)
# =====================================================================================

def evaluate(fen):
    """Utility U(s) for finished games, estimate E(s) otherwise (Session 15). White's view."""
    board = chess.Board(fen)
    if board.is_checkmate():                       # the side to move has been checkmated
        return -MATE if board.turn == chess.WHITE else MATE
    if game_is_over(board):                        # stalemate, insufficient material, fifty moves
        return 0                                   # a draw is worth 0 for both players
    return static_evaluation(board)


def minimax(fen, depth):
    """Plain minimax with a depth cutoff (Session 14 + Session 15). Visits every position."""
    board = chess.Board(fen)
    nodes = 0                                      # counts every position visited, root included

    def value(depth):
        nonlocal nodes                             # lets the inner function update the counter
        nodes += 1
        if depth == 0 or game_is_over(board):      # leaf: cutoff depth reached or game finished
            return evaluate(board.fen())           # handbook: score leaves with evaluate()
        results = []
        for move in board.legal_moves:             # ACTIONS(s)
            board.push(move)                       # RESULT(s, a): make the move ...
            results.append(value(depth - 1))
            board.pop()                            # ... and undo it, so the board is unchanged
        if board.turn == chess.WHITE:              # White to move = MAX node
            return max(results)
        return min(results)                        # Black to move = MIN node

    return {"value": value(depth), "nodes": nodes}


def alphabeta(fen, depth):
    """Minimax with alpha-beta pruning (Session 14). Same value as minimax, fewer nodes.
    alpha = best score MAX can already guarantee; beta = best score MIN can already guarantee."""
    board = chess.Board(fen)
    nodes = 0

    def value(depth, alpha, beta):
        nonlocal nodes
        nodes += 1
        if depth == 0 or game_is_over(board):
            return evaluate(board.fen())
        # Move ordering (Session 14): captures first, because they often cause early cut-offs.
        # sorted() puts False (0) before True (1), so "not a capture" sends quiet moves last.
        moves = sorted(board.legal_moves, key=lambda move: not board.is_capture(move))
        if board.turn == chess.WHITE:              # MAX node
            best = -INFINITY
            for move in moves:
                board.push(move)
                best = max(best, value(depth - 1, alpha, beta))
                board.pop()
                alpha = max(alpha, best)           # MAX's guarantee can only go up
                if alpha >= beta:                  # MIN already has something at least as good elsewhere,
                    break                          # so MIN will never let the game reach here: prune
            return best
        best = INFINITY                            # MIN node: mirror image of the MAX node
        for move in moves:
            board.push(move)
            best = min(best, value(depth - 1, alpha, beta))
            board.pop()
            beta = min(beta, best)                 # MIN's guarantee can only go down
            if alpha >= beta:                      # MAX will never allow this branch: prune
                break
        return best

    return {"value": value(depth, -INFINITY, INFINITY), "nodes": nodes}


# =====================================================================================
# PART 3. PLAYING SEARCH (what act() really uses)
# Same alpha-beta as above, plus four additions that make it fast enough to play:
#   1. Move ordering with MVV-LVA (Session 14, "move ordering determines pruning efficiency")
#   2. Iterative deepening (handbook optional extension): depth 1, 2, 3 ... until time runs out
#   3. Transposition table (handbook optional extension): remember positions already searched
#   4. Quiescence search: at the cutoff, keep following captures (fixes the horizon effect)
# =====================================================================================

class OutOfTime(Exception):
    """Signal that the time for this move has run out: check_clock() raises it in 
    the middle of the search, however deep the recursion is at that moment. That
    way we jump straight back to choose_move() without returning through every 
    level by hand. choose_move() catches it and plays the best move from the 
    deepest search that finished, or the partial result of the current
    depth when that is safe."""


EXACT, LOWER_BOUND, UPPER_BOUND = 0, 1, 2        # what kind of score a table entry holds


class SearchState:
    """Everything the search remembers. One object per game (the judge starts a new process each game)."""

    def __init__(self):
        self.table = {}          # transposition table: position hash -> (depth, value, flag, best move)
        self.seen = set()        # hashes of positions already in the game or on the current search line
        self.nodes = 0           # positions visited (used to check the clock every 64 nodes)
        self.deadline = 0.0      # time.perf_counter() value at which we must stop
        self.partial = None      # (move, score) of the depth being searched, updated after each root move


state = SearchState()            # module-level: survives between act() calls within one game


def position_key(board):
    """A 64-bit number identifying the position (pieces, side to move, castling, en passant)."""
    return chess.polyglot.zobrist_hash(board)


def mate_score(board, ply):
    """Score when the side to move is checkmated, `ply` half-moves below the root.
    Subtracting ply makes a quick mate worth more than a slow one, so the agent mates fastest."""
    if board.turn == chess.WHITE:        # White is mated: very bad for MAX
        return -(MATE - ply)
    return MATE - ply                    # Black is mated: very good for MAX


def order_moves(board, moves, first_move=None):
    """Best-looking moves first. Good ordering = more alpha-beta cut-offs (Session 14)."""
    def priority(move):
        if move == first_move:                        # best move from an earlier search: try it first
            return 1000000
        if board.is_capture(move):
            # MVV-LVA: Most Valuable Victim, Least Valuable Attacker.
            # Pawn takes queen (900*10 - 100) is tried before queen takes pawn (100*10 - 900).
            victim = board.piece_type_at(move.to_square) or chess.PAWN   # None only for en passant
            attacker = board.piece_type_at(move.from_square)
            return 10000 + PIECE_VALUE[victim] * 10 - PIECE_VALUE[attacker]
        if move.promotion == chess.QUEEN:             # promoting is almost as good as a capture
            return 9000
        return 0                                      # quiet moves last, in generation order
    return sorted(moves, key=priority, reverse=True)


def check_clock():
    """Count a node; every 64 nodes look at the clock (looking at every node would waste time)."""
    state.nodes += 1
    if state.nodes % 64 == 0 and time.perf_counter() > state.deadline:
        raise OutOfTime()


def is_losing_capture(board, move):
    """Spot captures that will most likely lose material, such as a queen taking a
    defended pawn:  A capture counts as losing when the capturing piece is worth 
    more than the piece it takes and the opponent can recapture on that square. 
    Quiescence search skips these moves to save time, because following them 
    rarely changes the result. The main search still looks at them, so a capture 
    that really works is not missed. Promotions are never treated as losing: they 
    are always worth a look."""
   
    if move.promotion:                                           # promotions are always worth a look
        return False
    victim = board.piece_type_at(move.to_square) or chess.PAWN   # None only for en passant (a pawn)
    attacker = board.piece_type_at(move.from_square)
    defended = board.is_attacked_by(not board.turn, move.to_square)   # can the opponent recapture there?
    return PIECE_VALUE[attacker] > PIECE_VALUE[victim] and defended


def quiescence(board, alpha, beta, ply):
    """Keep searching after the depth limit, but only captures and queen promotions,
    until the position is calm: The problem it solves: if the search stops at a fixed 
    depth, it can stop right in the middle of an exchange. Imagine the last move it 
    looks at is "queen takes pawn". It sees a pawn won and is happy, but it never sees 
    that next move the opponent takes the queen. Session 14 calls this the HORIZON 
    EFFECT: what lies just beyond the search horizon is invisible. The fix is to look 
    only at captures, which are few, until no more captures are worth making. Only 
    then do we trust static_evaluation(). Each side is also allowed to "stand pat", 
    that is, to stop capturing if capturing would make things worse. So the static 
    score acts as a floor, and captures are only followed when they can improve on it.
    When the side to move is in check it cannot just stand still, so we look at 
    every legal move instead of only captures."""

    check_clock()
    maximizing = board.turn == chess.WHITE

    if board.is_check():
        # In check you cannot "do nothing", so we must look at every way out of check.
        moves = list(board.legal_moves)
        if not moves:                                  # in check with no moves: checkmate
            return mate_score(board, ply)
        if ply > 60:                                   # safety limit on very long sequences
            return static_evaluation(board)
        best = -INFINITY if maximizing else INFINITY
    else:
        # "Stand pat": the side to move may decline every capture, so the static score
        # is already a guaranteed result for it. Captures are only searched to improve on it.
        stand_pat = static_evaluation(board)
        if ply > 60:
            return stand_pat
        if maximizing:
            if stand_pat >= beta:                      # already too good: MIN will avoid this position
                return stand_pat
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:                     # already too good for MIN: MAX will avoid it
                return stand_pat
            beta = min(beta, stand_pat)
        best = stand_pat
        moves = [move for move in board.legal_moves
                 if (board.is_capture(move) or move.promotion == chess.QUEEN)
                 and not is_losing_capture(board, move)]

    for move in order_moves(board, moves):
        board.push(move)
        score = quiescence(board, alpha, beta, ply + 1)
        board.pop()
        if maximizing:                                 # same alpha-beta update as in alphabeta()
            best = max(best, score)
            alpha = max(alpha, best)
        else:
            best = min(best, score)
            beta = min(beta, best)
        if alpha >= beta:
            break
    return best


def search(board, depth, alpha, beta, ply):
    """Alpha-beta for the playing agent. Same MAX/MIN logic as alphabeta(), with the additions."""
    check_clock()
    key = position_key(board)

    # 1. Draws by rule. A position that already appeared (in the game or on this search line)
    #    is scored as a draw: repeating it gains nothing, and threefold repetition is a draw.
    if key in state.seen:
        return 0
    if board.halfmove_clock >= 100 and not board.is_checkmate():      # fifty-move rule
        return 0
    if board.is_insufficient_material():                               # e.g. king against king
        return 0

    # 2. Depth cutoff reached: instead of evaluating immediately, settle the captures first.
    if depth <= 0:
        return quiescence(board, alpha, beta, ply)

    # 3. Transposition table: the same position can be reached by different move orders
    #    (e.g. Nf3 then e4, or e4 then Nf3). If we already searched it deeply enough, reuse it.
    table_move = None
    entry = state.table.get(key)
    if entry is not None:
        stored_depth, stored_value, flag, table_move = entry
        # Near the fifty-move limit the same position can have a different future, so don't trust it.
        if stored_depth >= depth and board.halfmove_clock < 90:
            if flag == EXACT:
                return stored_value
            if flag == LOWER_BOUND and stored_value >= beta:   # true value >= stored >= beta: cut-off
                return stored_value
            if flag == UPPER_BOUND and stored_value <= alpha:  # true value <= stored <= alpha: cut-off
                return stored_value

    # 4. Terminal test: no legal moves means checkmate or stalemate.
    moves = list(board.legal_moves)
    if not moves:
        return mate_score(board, ply) if board.is_check() else 0

    # 5. Ordinary alpha-beta over the ordered moves (same as alphabeta() in Part 2).
    maximizing = board.turn == chess.WHITE
    original_alpha, original_beta = alpha, beta          # needed to label the result for the table
    best = -INFINITY if maximizing else INFINITY
    best_move = None
    state.seen.add(key)                                  # this position is now on the current line
    try:
        for move in order_moves(board, moves, table_move):
            board.push(move)
            score = search(board, depth - 1, alpha, beta, ply + 1)
            board.pop()
            if maximizing and score > best:
                best, best_move = score, move
                alpha = max(alpha, best)
            elif not maximizing and score < best:
                best, best_move = score, move
                beta = min(beta, best)
            if alpha >= beta:                            # prune the remaining moves
                break
    finally:
        state.seen.discard(key)                          # leaving this position: remove it from the line

    # 6. Store the result. If we stopped early (cut-off) the score is only a bound, not exact.
    if best <= original_alpha:
        flag = UPPER_BOUND                               # every move was <= alpha: true value <= best
    elif best >= original_beta:
        flag = LOWER_BOUND                               # cut-off: true value >= best
    else:
        flag = EXACT
    # Mate scores depend on the distance to the root, and draw scores of 0 may come from a
    # repetition that depends on the path we took. We keep only the move for those (depth -1).
    stored_depth = depth if (best != 0 and abs(best) < MATE_THRESHOLD) else -1
    if len(state.table) > 300000:                        # keep memory far below the 2 GB limit
        state.table.clear()
    state.table[key] = (stored_depth, best, flag, best_move)
    return best


def search_root(board, moves, depth):
    """One complete alpha-beta search to `depth` from the current position. Returns (move, score)."""
    maximizing = board.turn == chess.WHITE
    alpha, beta = -INFINITY, INFINITY
    best_move, best = moves[0], (-INFINITY if maximizing else INFINITY)
    for move in moves:
        board.push(move)
        score = search(board, depth - 1, alpha, beta, 1)
        board.pop()
        if maximizing and score > best:
            best_move, best = move, score
            alpha = max(alpha, score)
        elif not maximizing and score < best:
            best_move, best = move, score
            beta = min(beta, score)
        state.partial = (best_move, best)       # best so far among the root moves FULLY searched at this depth
    return best_move, best


# =====================================================================================
# PART 4. THE AGENT: act(), clock management and the board's history
# =====================================================================================

RESERVE_S = 0.5        # handbook: keep about half a second for Colab's slower machines
PANIC_S = 0.25         # below this, answer instantly without searching (safety for slow machines)


def rebuild_board(observation):
    """A FEN has no history, so we replay all moves from the start (handbook advice).
    Also returns the hashes of the positions that could still repeat."""
    board = chess.Board()
    seen = [position_key(board)]
    try:
        for uci in observation["history"]:
            board.push_uci(uci)
            if board.halfmove_clock == 0:      # a capture or pawn move: older positions can never come back
                seen = []
            seen.append(position_key(board))
        if board.fen() == observation["fen"]:  # sanity check: the replay matches the judge's position
            return board, set(seen)
    except ValueError:                         # a history we cannot replay: fall back to the FEN
        pass
    board = chess.Board(observation["fen"])
    return board, {position_key(board)}


def time_budget(remaining, ply):
    """How long to think about this move: (target, hard limit) in seconds."""
    usable = max(0.0, remaining - RESERVE_S)
    moves_to_go = max(20, 36 - ply // 4)       # assume many moves left early, fewer later (never under 20)
    target = usable / moves_to_go              # share the clock across the expected remaining moves
    hard_limit = min(3 * target, 0.15 * usable)  # never spend more than 15% of what is left on one move
    return target, hard_limit


def choose_move(observation):
    started = time.perf_counter()
    board, seen = rebuild_board(observation)
    state.seen = seen
    target, hard_limit = time_budget(observation["time_remaining_s"], observation["ply"])
    state.deadline = started + hard_limit

    moves = order_moves(board, list(board.legal_moves))
    best_move = moves[0]                       # always have an answer, even if depth 1 is not finished
    depth_done, score = 0, 0
    # Iterative deepening: search depth 1, then 2, then 3 ... Each finished depth gives a usable
    # move, so when the time runs out we play the best move of the deepest COMPLETE search.
    for depth in range(1, 50):
        state.partial = None
        try:
            move, score = search_root(board, moves, depth)
        except OutOfTime:
            # The depth is unfinished. moves[0] is the previous best move and was searched first,
            # so if at least that move finished, state.partial is the best move at this new depth
            # among the moves already examined: at least as reliable as the previous answer.
            if state.partial is not None and depth > 1:
                best_move = state.partial[0]
            break
        best_move, depth_done = move, depth
        moves.remove(move)                     # put the best move first for the next, deeper search:
        moves.insert(0, move)                  # better ordering means more cut-offs (Session 14)
        if abs(score) >= MATE_THRESHOLD:       # a forced mate (for or against us) was found: deeper won't help
            break
        if time.perf_counter() - started > 0.5 * target:
            break                              # the next depth takes several times longer: don't start it
    print(f"depth={depth_done} score={score} time={time.perf_counter() - started:.3f}s")
    return best_move.uci()


def act(observation):
    legal = observation["legal_moves"]
    if len(legal) == 1:                        # only one legal move: no need to think
        return {"move": legal[0]}
    try:
        if observation["time_remaining_s"] < PANIC_S:
            board = chess.Board(observation["fen"])
            moves = order_moves(board, list(board.legal_moves))
            return {"move": moves[0].uci()}    # almost no time: best-looking capture, no search
        return {"move": choose_move(observation)}
    except Exception as error:                 # a bug must never lose the game: play a legal move
        print("ERROR, fallback move used:", repr(error))
        return {"move": legal[0]}