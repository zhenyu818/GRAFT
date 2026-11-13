FROM scratch
COPY bin/graft-manager /manager
USER 10001:10001
ENTRYPOINT ["/manager"]
