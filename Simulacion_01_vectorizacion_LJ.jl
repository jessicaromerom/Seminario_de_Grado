using Random
using Plots
using DataFrames
using CSV
using Statistics
using LinearAlgebra

#Parametros Simulacion ABP
n = 400
L = 100
Dr = 0.0225
Dt = 0.03
v0 = 20
dt = 0.001
steps = 5000
tend = steps * dt
a = 1
e = 3
sigma = 2 * a
r_min = 2^(1 / 6) * sigma
frec_save = 10

incremento_rotacional = sqrt(2 * Dr * dt)
incremento_traslacional = sqrt(2 * Dt * dt)

#Angulos iniciales
theta = 2 * pi * rand(n)
cost = cos.(theta)
sint = sin.(theta)

#Velocidades iniciales
velocidades = [v0 * cost, v0 * sint]

#Parametros Dinamica SIR
#Colores y radio de contagio
colores = fill("blue", n)
colores[1] = "red"
R = 1.5 * r_min
#SIR
susceptibles = fill(1, n)
susceptibles[1] = 0
infectados = fill(0, n)
infectados[1] = 1
recuperados = fill(0, n)
#Recuperacion
tmin = 500
tmax = 750
tiempo_rec = (tmax - tmin) * rand(n) .+ tmin
contador_tiempo = zeros(n)

paso_tiempo = convert(Int, (steps / frec_save) + 1)

"""Funcion de posiciones_iniciales!, LJ_fuerza!, dinamica_ABP!"""
function posiciones_iniciales!(n, sigma, L)
    min = -L / 2 + sigma
    max = L / 2 - sigma

    posiciones = zeros(n, 2)
    X = 0
    Y = 0

    for i in 1:n
        valid_posicion = false
        while !valid_posicion
            X = (max - min) * rand() + min
            Y = (max - min) * rand() + min
            valid_posicion = true

            for j in 1:i
                distance = sqrt((X - posiciones[j, 1])^2 + (Y - posiciones[j, 2])^2)
                if distance < sigma * 2.0
                    valid_posicion = false
                    break
                end
            end
        end

        posiciones[i, 1] = X
        posiciones[i, 2] = Y
    end
    return posiciones
end

Dtensor = ones(n,n)
I = ones(n)

function LJ_fuerza!(X, Y, Dtensor, I, L, sigma, e, r_min)
    #se puede aplicar el broadcasting @. (?)
    dx = X .- transpose(X)
    dy = Y .- transpose(Y)

    dx .-= L * round.(dx / L)
    dy .-= L * round.(dy / L)

    Distance_square = dx .* dx + dy .* dy
    D = sqrt.(Distance_square)
    inv_12 = 1 ./ (D.^12)
    inv_6 = 1 ./ (D.^6)

    id = (D .== 0)
    inv_12[id] .= 0
    inv_6[id] .= 0

    id_larger_r_min = D .> r_min
    D_copy = copy(D)
    D_copy[id_larger_r_min] .= 0

    force_particles = -4 * e * (12 * sigma^12 .* (Dtensor .* inv_12) .- 6 * sigma^6 .* (Dtensor .* inv_6))

    Fx = force_particles .* dx
    Fy = force_particles .* dy

    Fx = Fx * I
    Fy = Fy * I

    return Fx, Fy
end


function dinamica_ABP!(X, Y, velocidades)

    #Cambio en el angulo de las particulas
    @. theta += incremento_rotacional * randn() 

    velocidades[1] .= v0 * cos.(theta)
    velocidades[2] .= v0 * sin.(theta)

    #calculo de LJ_fuerza
    Fx, Fy = LJ_fuerza!(X, Y, Dtensor, I, L, sigma, e, r_min)

    #separar el calculo de las posiciones(?) agregar incremento_traslacional(?)
    @. X += velocidades[1] * dt + Fx * dt 
    @. Y += velocidades[2] * dt + Fy * dt 

    #Condiciones de borde periodicas 
    @. X = mod(X + L / 2, L) - L / 2
    @. Y = mod(Y + L / 2, L) - L / 2

    #Dinamica de contagio SIR
    #Susceptibles -> Infectados
    for i in 1:n, j in 1:n
        dx, dy = X[i] - X[j], Y[i] - Y[j]
        distancia = sqrt(dx^2 + dy^2)
        if distancia < R && colores[i] == "blue" && colores[j] == "red"
            colores[i] = "red"
            infectados[i] = 1
            susceptibles[i] = 0
            if distancia < R && colores[j] == "blue" && colores[i] == "red"
                colores[j] = "red"
                infectados[j] = 1
                susceptibles[j] = 0
            end
        end
    end

    # Infectados -> Recuperados
    for i in 1:n
        if colores[i] == "red"
            contador_tiempo[i] += 1 #tiempo desde que fue infectada 
            if colores[i] == "red" && contador_tiempo[i] > tiempo_rec[i]
                colores[i] = "green"
                infectados[i] = 0
                recuperados[i] = 1
                contador_tiempo[i] = 0
            end
        end
    end

    return X, Y, colores, susceptibles, infectados, recuperados
end

posiciones = posiciones_iniciales!(n, sigma, L)
X = posiciones[:, 1]
Y = posiciones[:, 2]

posiciones_x = Vector{Vector{Float64}}()
posiciones_y = Vector{Vector{Float64}}()
color = Vector{Vector{String}}()
historial_susceptibles = []
historial_infectados = []
historial_recuperados = []

@time for i in 1:steps
    dinamica_ABP!(X, Y, velocidades)
    susceptibles_actual = sum(susceptibles)
    infectados_actual = sum(infectados)
    recuperados_actual = sum(recuperados)
    if i == 1 || mod(i, frec_save) == 0
        push!(posiciones_x, copy(X))
        push!(posiciones_y, copy(Y))
        push!(color, copy(colores))
        push!(historial_susceptibles, susceptibles_actual)
        push!(historial_infectados, infectados_actual)
        push!(historial_recuperados, recuperados_actual)
        end
end

#CSV.write("posiciones_particulas.csv", DataFrame(X = posiciones_x, Y = posiciones_y))
CSV.write("posiciones_particulas.csv", DataFrame(X = posiciones_x, Y = posiciones_y, Colores = color))
#Curvas de contagio
CSV.write("Dinamica_SIR.csv", DataFrame(S = historial_susceptibles, I = historial_infectados, R = historial_recuperados))

