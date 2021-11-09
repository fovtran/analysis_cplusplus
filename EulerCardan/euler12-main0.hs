-module (euler12).
-compile (export_all).

factorCount (Number) -> factorCount (Number, math:sqrt (Number), 1, 0).

factorCount (_, Sqrt, Candidate, Count) when Candidate > Sqrt -> Count;

factorCount (_, Sqrt, Candidate, Count) when Candidate == Sqrt -> Count + 1;

factorCount (Number, Sqrt, Candidate, Count) ->
    case Number rem Candidate of
        0 -> factorCount (Number, Sqrt, Candidate + 1, Count + 2);
        _ -> factorCount (Number, Sqrt, Candidate + 1, Count)
    end.

nextTriangle (Index, Triangle) ->
    Count = factorCount (Triangle),
    if
        Count > 1000 -> Triangle;
        true -> nextTriangle (Index + 1, Triangle + Index + 1)
    end.

solve () ->
    io:format ("~p~n", [nextTriangle (1, 1) ] ),
    halt (0).
