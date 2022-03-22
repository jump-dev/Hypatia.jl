
insts = OrderedDict()
insts["minimal"] = [
    # nonsymmetric:
    ((false, false, false, 3, 3, 0.8),),
    ((false, false, false, 3, 3, 0.8), :ExpPSD),
    ((true, false, false, 2, 4, 0.7),),
    ((true, false, false, 2, 4, 0.7), :SOCExpPSD),
    # symmetric:
    ((false, true, false, 3, 3, 0.4),),
    ((false, true, true, 3, 3, 0.4), :ExpPSD),
    ((true, true, false, 4, 4, 0.7),),
    ((true, true, true, 4, 4, 0.7), :SOCExpPSD),
    ]
insts["fast"] = [
    # nonsymmetric:
    ((false, false, false, 4, 12, 0.5),),
    ((false, false, false, 4, 12, 0.5), :ExpPSD),
    ((true, false, false, 5, 10, 0.7),),
    ((true, false, false, 5, 10, 0.7), :SOCExpPSD),
    # symmetric:
    ((false, true, false, 15, 15, 0.5),),
    ((false, true, true, 15, 15, 0.5), :ExpPSD),
    ((true, true, false, 20, 20, 0.7),),
    ((true, true, true, 20, 20, 0.7), :SOCExpPSD),
    ]
insts["various"] = [
    # # nonsymmetric:
    # ((false, false, false, 20, 50, 0.8),),
    # ((false, false, false, 8, 20, 0.8), :ExpPSD),
    # ((true, false, false, 40, 50, 0.7),),
    # ((true, false, false, 15, 20, 0.7), :SOCExpPSD),
    # # symmetric:
    # ((false, true, false, 100, 100, 0.9),),
    # ((false, true, true, 100, 100, 0.9),),
    # ((true, true, false, 50, 50, 0.5),),
    # ((true, true, true, 10, 10, 0.5), :SOCExpPSD),
    # compile:
    ((false, false, false, 5, 10 * 5, 0.8),),
    # runs:
    ((false, false, false, 5, 10 * 5, 0.8),),
    ((false, false, false, 10, 10 * 10, 0.8),),
    ((false, false, false, 15, 10 * 15, 0.8),),
    ((false, false, false, 20, 10 * 20, 0.8),),
    ((false, false, false, 25, 10 * 25, 0.8),),
    ((false, false, false, 30, 10 * 30, 0.8),),
    ((false, false, false, 35, 10 * 35, 0.8),),
    ((false, false, false, 40, 10 * 40, 0.8),),
    ((false, false, false, 45, 10 * 45, 0.8),),
    ((false, false, false, 50, 10 * 50, 0.8),),
    ]
return (MatrixCompletionJuMP, insts)
