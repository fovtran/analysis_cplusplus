factorCount number = factorCount' number isquare 1 0 - (fromEnum $ square == fromIntegral isquare)
    where square = sqrt $ fromIntegral number
          isquare = floor square

factorCount' number sqrt candidate count
    | fromIntegral candidate > sqrt = count
    | number `mod` candidate == 0 = factorCount' number sqrt (candidate + 1) (count + 2)
    | otherwise = factorCount' number sqrt (candidate + 1) count

nextTriangle index triangle
    | factorCount triangle > 1000 = triangle
    | otherwise = nextTriangle (index + 1) (triangle + index + 1)

main = print $ nextTriangle 1 1
