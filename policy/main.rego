package modelpassport.deployment

import rego.v1

default allow := false

allow if {
    input.passport.status == "verified"
    input.passport.expired == false
    count(input.passport.blocking_findings) == 0
}

