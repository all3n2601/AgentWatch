package main

import (
	"flag"
	"fmt"
	"os"
)

var version = "dev"

func main() {
	flag.Parse()
	if flag.NArg() == 1 && flag.Arg(0) == "version" {
		fmt.Println(version)
		return
	}

	fmt.Fprintln(os.Stderr, "usage: passport version")
	os.Exit(2)
}
