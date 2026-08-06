function digest = sha256_file(file_path)
%SHA256_FILE Return a lowercase SHA-256 digest for one file.

assert(isfile(file_path), 'Cannot hash missing file: %s', file_path);
message_digest = java.security.MessageDigest.getInstance('SHA-256');
file_id = fopen(file_path, 'rb');
assert(file_id >= 0, 'Could not open file for hashing: %s', file_path);
cleanup = onCleanup(@() fclose(file_id)); %#ok<NASGU>
while true
    buffer = fread(file_id, 1024 * 1024, '*uint8');
    if isempty(buffer)
        break;
    end
    message_digest.update(typecast(buffer(:), 'int8'));
end
raw = typecast(message_digest.digest(), 'uint8');
digest = lower(reshape(dec2hex(raw, 2).', 1, []));
end
