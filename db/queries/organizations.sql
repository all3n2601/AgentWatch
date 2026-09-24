-- name: GetOrganization :one
SELECT id, name, created_at
FROM organizations
WHERE id = $1;

