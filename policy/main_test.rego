package modelpassport.deployment

import rego.v1

test_verified_passport_is_allowed if {
    allow with input as {
        "passport": {
            "status": "verified",
            "expired": false,
            "blocking_findings": [],
        },
    }
}

test_draft_passport_is_denied if {
    not allow with input as {
        "passport": {
            "status": "draft",
            "expired": false,
            "blocking_findings": [],
        },
    }
}

