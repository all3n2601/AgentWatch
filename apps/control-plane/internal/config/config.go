package config

import "os"

type Config struct {
	Address string
}

func FromEnvironment() Config {
	address := os.Getenv("CONTROL_PLANE_ADDRESS")
	if address == "" {
		address = ":8080"
	}

	return Config{Address: address}
}
