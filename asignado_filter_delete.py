class AsignadoFilter(admin.SimpleListFilter):
    _title = 'Asignación'
    parameter_name = 'asignado'

    def lookups(self, request, model_admin):
        return (
            ('sí', 'sí'),
            ('no', 'no'),
        )

    def queryset(self, request, queryset):
        value = self.value()

    @property
    def title(self):
        return self.title
    