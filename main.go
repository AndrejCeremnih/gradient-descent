package main

import (
	"fmt"
	"image"
	"log"
	"math/rand"

	"github.com/hajimehoshi/ebiten/v2"
	"gonum.org/v1/plot/plotter"
)

const (
	screenWidth, screenHeight                     = 640, 480
	randMin, randMax                              = -10, 10
	epochs, lr                                    = 1000000, 0.0001
	plotMinX, plotMaxX, plotMinY, plotMaxY        = -10, 10, -50, 100
	inputPointsMinY, inputPointsMaxY, inputPoints = -20, 20, 10
)

// Function points are spawed along
func f(x float64) float64 {
	return x*x + 5*x - 3
}

// func df(f func(float64) float64) func(float64) float64 {
// 	return func(x float64) float64 {
// 		dx := 1e-10 // dx -> 0
// 		return (f(x+dx) - f(x)) / dx
// 	}
// }

// Inference for 1 argument
func i(x, w, b float64) float64 { return w*x + b }

// For all the input data
func inference(x []float64, w, b float64) (out []float64) {
	for _, v := range x {
		out = append(out, i(v, w, b))
	}
	return
}

// func loss(y, labels []float64) float64 {
// 	var errSum float64
// 	for i := range labels {
// 		errSum += math.Pow((y[i] - labels[i]), 2)
// 	}
// 	return errSum / float64(len(labels)) // n
// }

func gradient(xs, ys, labels []float64) (w, b float64) {
	for i := 0; i < len(labels); i++ {
		dif := ys[i] - labels[i]
		w += dif * xs[i]
		b += dif
	}
	n := float64(len(labels))
	w *= 2 / n
	b *= 2 / n
	return
}

func train(inputs, labels []float64) (w, b float64) {
	randFloat64 := func() float64 {
		return randMin + rand.Float64()*(randMax-randMin)
	}
	w, b = randFloat64(), randFloat64()
	var dw, db float64
	for i := 0; i < epochs; i++ {
		dw, db = gradient(labels, inference(inputs, w, b), inputs)
		w -= dw * lr
		b -= db * lr
		fmt.Println(w, b)
	}
	return
}

func randPoints(f func(float64) float64, inputPointsMinY, inputPointsMaxY float64, inputPoints int) (xs, labels []float64) {
	for i := 0; i < inputPoints; i++ {
		x := plotMinX + (plotMaxX-plotMinX)*rand.Float64()
		inputPointsY := inputPointsMinY + (inputPointsMaxY-inputPointsMinY)*rand.Float64()
		xs = append(xs, x)
		labels = append(labels, f(x)+inputPointsY)
	}
	return
}

func main() {
	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Gradient descent")

	img := make(chan *image.RGBA, 1)
	inputs, labels := randPoints(f, inputPointsMinY, inputPointsMaxY, inputPoints)
	var points plotter.XYs
	for i := 0; i < inputPoints; i++ {
		points = append(points, plotter.XY{X: inputs[i], Y: labels[i]})
	}
	pointsScatter, _ := plotter.NewScatter(points)
	fp := plotter.NewFunction(f)
	w, b := train(inputs, labels)
	fmt.Println(w, b)
	ap := plotter.NewFunction(func(x float64) float64 { return w*x + b })
	img <- Plot(pointsScatter, fp, ap)
	if err := ebiten.RunGame(&App{Img: img}); err != nil {
		log.Fatal(err)
	}
}
