package main

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"

	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/linear"
	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"

	mnist "github.com/petar/GoMNIST"
)

func main() {
	set, err := mnist.ReadSet("./datasets/mnist/images.gz", "./datasets/mnist/labels.gz")
	if err != nil {
		panic(err)
	}

	// https://github.com/petar/GoMNIST/blob/master/mnist.go
	// from here we see that this is type RawImage []byte
	rawImage := set.Images[1]
	fmt.Println(reflect.TypeOf(rawImage))
	fmt.Println(rawImage) // GoMNIST.RawImage
	fmt.Println("-----")

	df := MNISTSetToDataframe(set, 1000)
	//categories := []string{"tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "shoe", "bag", "boot"}
	training, validation := Split(df, 0.75)

	trainingIsTrouser, err1 := EqualsInt(training.Col("Label"), 1)
	validationIsTrouser, err2 := EqualsInt(validation.Col("Label"), 1)
	if err1 != nil || err2 != nil {
		fmt.Println("Error", err1, err2)
	}

	trainingImages := ImageSeriesToFloats(training, "Image")
	validationImages := ImageSeriesToFloats(validation, "Image")

	model := linear.NewLogistic(base.BatchGA, 1e-4, 1, 150, trainingImages, trainingIsTrouser.Float())
	//Train
	err = model.Learn()
	if err != nil {
		fmt.Println(err)
	}

	var correct = 0.
	for i := range validationImages {
		prediction, err := model.Predict(validationImages[i])
		if err != nil {
			panic(err)
		}

		if math.Round(prediction[0]) == validationIsTrouser.Elem(i).Float() {
			correct++
		}
	}

	accuracy := correct / float64(len(validationImages))
	fmt.Println(accuracy) // 0.988, but it's pretty slow
}

func MNISTSetToDataframe(st *mnist.Set, maxExamples int) dataframe.DataFrame {
	length := maxExamples
	if length > len(st.Images) {
		length = len(st.Images)
	}
	s := make([]string, length, length)
	l := make([]int, length, length)
	for i := 0; i < length; i++ {
		s[i] = string(st.Images[i])
		l[i] = int(st.Labels[i])
	}
	var df dataframe.DataFrame
	images := series.Strings(s)
	images.Name = "Image"
	labels := series.Ints(l)
	labels.Name = "Label"
	df = dataframe.New(images, labels)
	return df
}

func Split(df dataframe.DataFrame, valFraction float64) (training dataframe.DataFrame, validation dataframe.DataFrame) {
	perm := rand.Perm(df.Nrow())
	cutoff := int(valFraction * float64(len(perm)))
	training = df.Subset(perm[:cutoff])
	validation = df.Subset(perm[cutoff:])
	return training, validation
}

func EqualsInt(s series.Series, to int) (*series.Series, error) {
	eq := make([]int, s.Len(), s.Len())
	ints, err := s.Int()
	if err != nil {
		return nil, err
	}
	for i := range ints {
		if ints[i] == to {
			eq[i] = 1
		}
	}
	ret := series.Ints(eq)
	return &ret, nil
}

func NormalizeBytes(bs []byte) []float64 {
	ret := make([]float64, len(bs), len(bs))
	for i := range bs {
		ret[i] = float64(bs[i]) / 255.
	}
	return ret
}

func ImageSeriesToFloats(df dataframe.DataFrame, col string) [][]float64 {
	s := df.Col(col)
	ret := make([][]float64, s.Len(), s.Len())
	for i := 0; i < s.Len(); i++ {
		b := []byte(s.Elem(i).String())
		ret[i] = NormalizeBytes(b)
	}
	return ret
}
